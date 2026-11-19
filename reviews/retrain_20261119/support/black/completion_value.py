"""One-shot absorbing-goal compensation derived from dense-return upper bound."""

import math

import torch

BONUS = 12.0
CONFIG = dict(
  version=1,
  completion_supplement=BONUS,
  reason="6*0.02/(1-0.99) maximum discounted common positive tail",
  original_terminal_and_rewards_preserved=True,
)


def verify_return_bound(cfg, rl):
  positive = {
    name: term.weight
    for name, term in cfg.rewards.items()
    if term.weight > 0 and not name.startswith("specialty_")
  }
  if positive != dict(
    track_linear_velocity=2.0, track_angular_velocity=2.0, upright=1.0, pose=1.0
  ):
    raise ValueError("Completion bound requires the verified four exp<=1 rewards")
  dt = cfg.sim.mujoco.timestep * cfg.decimation
  if not math.isclose(sum(positive.values()) * dt / (1 - rl.algorithm.gamma), BONUS):
    raise ValueError("Dense bound no longer matches the admitted control/PPO rate")
  return dict(
    maximum_positive_rate=6.0,
    step_dt=dt,
    gamma=rl.algorithm.gamma,
    discounted_common_upper_bound=BONUS,
  )


class CompletionValueLedger:
  fields = ("paid", "owed", "blocked")

  def __init__(self, n, device):
    self.paid = torch.zeros(n, dtype=torch.bool, device=device)
    self.owed = torch.zeros_like(self.paid)
    self.blocked = torch.zeros_like(self.paid)
    self.last_step = -1
    self.paid_step = -1
    self.dt = 0.02
    self.migration = None

  def prepare(self, step, dt, *, active, finished, failed, valid):
    if step == self.last_step:
      return
    if step < self.last_step or not 0 < dt < 1:
      raise ValueError("Invalid completion payment clock")
    self.last_step, self.dt = step, dt
    invalid = active & (failed | ~valid)
    self.blocked |= invalid
    self.owed[invalid] = False
    self.owed |= active & finished & ~failed & valid & ~self.blocked & ~self.paid

  def consume(self, step):
    if step != self.last_step:
      raise RuntimeError("Original strict success snapshot must precede payment")
    if step == self.paid_step:
      return torch.zeros_like(self.paid, dtype=torch.float32)
    reward = self.owed.float() * (BONUS / self.dt)
    self.paid |= self.owed
    self.owed.zero_()
    self.paid_step = step
    return reward

  def reset(self, ids):
    for name in self.fields:
      getattr(self, name)[ids] = False

  def state_dict(self):
    return dict(
      config=CONFIG,
      last_step=self.last_step,
      paid_step=self.paid_step,
      dt=self.dt,
      migration=self.migration,
      tensors={k: getattr(self, k).detach().cpu().clone() for k in self.fields},
    )

  def load_state_dict(self, saved):
    if saved is None:
      self.blocked.fill_(True)
      self.migration = "No retroactive completion top-up; withheld until fresh reset"
      return
    if saved["config"] != CONFIG:
      raise ValueError("Completion compensation schema mismatch")
    for name in self.fields:
      value, target = saved["tensors"][name], getattr(self, name)
      if value.shape != target.shape or value.dtype != target.dtype:
        raise ValueError("Invalid completion tensor " + name)
      target.copy_(value.to(target.device))
    if (self.paid & self.owed).any() or (self.blocked & self.owed).any():
      raise ValueError("Completion cannot be paid twice or owed while blocked")
    self.last_step, self.paid_step = saved["last_step"], saved["paid_step"]
    self.dt, self.migration = saved["dt"], saved.get("migration")
