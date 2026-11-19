"""Once-only shaping for progress across the full strict exit hold horizon."""

import torch

CAP_TOTAL = 1.0
DEFAULT_SETTLE_STEPS = 50


def make_config(settle_steps: int = DEFAULT_SETTLE_STEPS) -> dict:
  partial_steps = max(1, int(settle_steps) - 1)
  return dict(
    version=2,
    per_new_best_strict_hold_step=CAP_TOTAL / partial_steps,
    maximum_rewarded_partial_steps=partial_steps,
    per_episode_cap=CAP_TOTAL,
    require_original_strict_gates=True,
    success_and_curriculum_unchanged=True,
  )


CONFIG = make_config(DEFAULT_SETTLE_STEPS)


class ExitStabilityLedger:
  fields = ("best", "paid_total", "owed_total", "blocked")

  def __init__(self, n, device, settle_steps: int = DEFAULT_SETTLE_STEPS):
    self.partial_steps = max(1, int(settle_steps) - 1)
    self.step_reward = CAP_TOTAL / self.partial_steps
    self.config = make_config(settle_steps)
    self.best = torch.zeros(n, dtype=torch.long, device=device)
    self.paid_total = torch.zeros(n, dtype=torch.float32, device=device)
    self.owed_total = torch.zeros(n, dtype=torch.float32, device=device)
    self.blocked = torch.zeros(n, dtype=torch.bool, device=device)
    self.last_step = -1
    self.paid_step = -1
    self.dt = 0.02
    self.migration = None

  def prepare(self, step, dt, *, original_stable_steps, active, failed, valid):
    if step == self.last_step:
      return
    if step < self.last_step or not 0 < dt < 1:
      raise ValueError("Invalid exit shaping clock")
    self.last_step, self.dt = step, dt
    invalid = active & (failed | ~valid)
    self.blocked |= invalid
    self.owed_total[invalid] = 0.0
    live = active & ~self.blocked
    # Smooth monotone shaping across 1..(settle_steps-1), never pays twice.
    candidate = original_stable_steps.clamp(min=0, max=self.partial_steps)
    increment = (candidate - self.best).clamp(min=0)
    increment = torch.where(live, increment, 0)
    self.best += increment
    self.owed_total += increment.float() * self.step_reward

  def consume(self, step):
    if step != self.last_step:
      raise RuntimeError("Original terminal snapshot must precede exit shaping")
    if step == self.paid_step:
      return torch.zeros_like(self.paid_total)
    reward = self.owed_total / self.dt
    self.paid_total += self.owed_total
    self.owed_total.zero_()
    self.paid_step = step
    return reward

  def reset(self, ids):
    for name in self.fields:
      getattr(self, name)[ids] = 0

  def state_dict(self):
    return dict(
      version=2,
      config=self.config,
      last_step=self.last_step,
      paid_step=self.paid_step,
      dt=self.dt,
      migration=self.migration,
      tensors={k: getattr(self, k).detach().cpu().clone() for k in self.fields},
    )

  def load_state_dict(self, saved):
    if saved is None:
      self.blocked.fill_(True)
      self.migration = "withheld_until_fresh_physical_reset"
      return
    if saved["version"] != 2 or saved["config"] != self.config:
      raise ValueError("Exit shaping schema/config mismatch")
    for name in self.fields:
      source = saved["tensors"][name]
      target = getattr(self, name)
      if source.shape != target.shape or source.dtype != target.dtype:
        raise ValueError("Invalid exit shaping tensor " + name)
      target.copy_(source.to(target.device))
    self.last_step = saved["last_step"]
    self.paid_step = saved["paid_step"]
    self.dt = saved["dt"]
    self.migration = saved.get("migration")
    if (
      (self.best < 0).any()
      or (self.best > self.partial_steps).any()
      or (self.paid_total < -1e-6).any()
      or (self.owed_total < -1e-6).any()
      or ((self.paid_total + self.owed_total) > CAP_TOTAL + 1e-5).any()
    ):
      raise ValueError("Invalid exit shaping budget")
