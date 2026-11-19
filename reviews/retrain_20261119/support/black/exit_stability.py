"""Once-only shaping for progress toward the unchanged six-step exit hold."""

import torch

STEP_MILLI = 25
PARTIAL_STEPS = 5
CAP_MILLI = STEP_MILLI * PARTIAL_STEPS
CONFIG = dict(
  per_new_best_strict_hold_step=0.025,
  maximum_rewarded_partial_steps=PARTIAL_STEPS,
  per_episode_cap=0.125,
  require_original_strict_gates=True,
  success_and_curriculum_unchanged=True,
)


class ExitStabilityLedger:
  fields = ("best", "paid_milli", "owed_milli", "blocked")

  def __init__(self, n, device):
    self.best = torch.zeros(n, dtype=torch.long, device=device)
    self.paid_milli = torch.zeros_like(self.best)
    self.owed_milli = torch.zeros_like(self.best)
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
    self.owed_milli[invalid] = 0
    live = active & ~self.blocked
    # This counter already requires all physical/safety/stability gates.
    # Repeated one-frame contacts or repeated streaks never earn twice.
    candidate = original_stable_steps.clamp(min=0, max=PARTIAL_STEPS)
    increment = (candidate - self.best).clamp(min=0)
    increment = torch.where(live, increment, 0)
    self.best += increment
    self.owed_milli += increment * STEP_MILLI

  def consume(self, step):
    if step != self.last_step:
      raise RuntimeError("Original terminal snapshot must precede exit shaping")
    if step == self.paid_step:
      return torch.zeros_like(self.best, dtype=torch.float32)
    reward = self.owed_milli.float() / (1000 * self.dt)
    self.paid_milli += self.owed_milli
    self.owed_milli.zero_()
    self.paid_step = step
    return reward

  def reset(self, ids):
    for name in self.fields:
      getattr(self, name)[ids] = 0

  def state_dict(self):
    return dict(
      version=1,
      config=CONFIG,
      last_step=self.last_step,
      paid_step=self.paid_step,
      dt=self.dt,
      migration=self.migration,
      tensors={k: getattr(self, k).detach().cpu().clone() for k in self.fields},
    )

  def load_state_dict(self, saved):
    if saved is None:
      # No inference from an old episode or its position can pay new credit.
      self.blocked.fill_(True)
      self.migration = "withheld_until_fresh_physical_reset"
      return
    if saved["version"] != 1 or saved["config"] != CONFIG:
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
      or (self.best > PARTIAL_STEPS).any()
      or (self.paid_milli < 0).any()
      or (self.owed_milli < 0).any()
      or ((self.paid_milli + self.owed_milli) > self.best * STEP_MILLI).any()
    ):
      raise ValueError("Invalid exit shaping budget")
