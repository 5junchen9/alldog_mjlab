"""Mixed continuation sampling, isolated from all existing Black tasks.

Bucket IDs are stable for downstream teacher masks and coverage reports. Transition
kind is -1 outside that bucket, 0 reversal, 1 stop, 2 start. A start deliberately
sets its source command to zero; physical robot state is never changed.
"""

from dataclasses import dataclass

import torch

from .omni_command import OmniVelocityCommand, OmniVelocityCommandCfg

BUCKET_NAMES = (
  "low_pure",
  "low_mixed",
  "high_pure",
  "medium_mixed",
  "boundary_mixed",
  "transition",
  "stand",
)
STAGE_WEIGHTS = {
  "stage1": (0.45, 0.15, 0.10, 0.10, 0.0, 0.10, 0.10),
  "stage2": (0.15, 0.10, 0.25, 0.20, 0.05, 0.15, 0.10),
}


def sample_targets(stage: str, count: int, device: str):
  """Return targets and buckets; transition targets are provisional start targets."""
  if stage not in STAGE_WEIGHTS:
    raise ValueError("stage must be stage1 or stage2")
  bucket = torch.multinomial(
    torch.tensor(STAGE_WEIGHTS[stage], device=device), count, replacement=True
  )
  targets = torch.zeros(count, 3, device=device)
  # Six signed axes are equiprobable, including for transition start targets.
  direction = torch.randint(6, (count,), device=device)
  axis = direction // 2
  sign = 1 - 2 * (direction % 2)
  high = bucket == 2
  lo = torch.where(high, torch.where(axis == 2, 0.3, 0.5), 0.05)
  hi = torch.where(high, torch.where(axis == 2, 0.5, 1.0), 0.30)
  magnitude = lo + torch.rand(count, device=device) * (hi - lo)
  targets[torch.arange(count, device=device), axis] = magnitude * sign
  low_mixed = bucket == 1
  angle = torch.rand(count, device=device) * (2 * torch.pi)
  speed = 0.10 + 0.40 * torch.rand(count, device=device)
  targets[low_mixed, 0] = (speed * angle.cos())[low_mixed]
  targets[low_mixed, 1] = (speed * angle.sin())[low_mixed]
  targets[low_mixed, 2] = (torch.rand(count, device=device) * 0.6 - 0.3)[low_mixed]
  signs = torch.randint(2, (count, 3), device=device) * 2 - 1
  mixed = torch.rand(count, 3, device=device)
  mixed[:, :2] = 0.3 + 0.5 * mixed[:, :2]
  mixed[:, 2] = 0.1 + 0.3 * mixed[:, 2]
  targets[bucket == 3] = (mixed * signs)[bucket == 3]
  targets[bucket == 4] = (signs * torch.tensor([1.0, 1.0, 0.5], device=device))[
    bucket == 4
  ]
  targets[bucket == 6] = 0
  return targets, bucket


class ContinuationVelocityCommand(OmniVelocityCommand):
  def __init__(self, cfg: "ContinuationVelocityCommandCfg", env):
    super().__init__(cfg, env)
    self.continuation_cfg = cfg
    self.target_command = torch.zeros_like(self.command)
    self.task_class = torch.full((self.num_envs,), 6, device=self.device)
    self.transition_kind = torch.full_like(self.task_class, -1)
    self.transition_age = torch.zeros(self.num_envs, device=self.device)
    self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.abrupt = torch.ones_like(self.initialized)
    self.sample_counts = torch.zeros(7, dtype=torch.long, device=self.device)
    self._command_dt = torch.full_like(self.transition_age, env.step_dt)

  @property
  def current_command(self):
    return self.command

  def reset(self, env_ids):
    self.initialized[env_ids] = False
    self.vel_command_b[env_ids] = 0
    self.target_command[env_ids] = 0
    return super().reset(env_ids)

  def _resample_command(self, env_ids):
    target, bucket = sample_targets(
      self.continuation_cfg.stage, len(env_ids), self.device
    )
    previous = self.command[env_ids].clone()
    transition = bucket == 5
    eligible = self.initialized[env_ids] & (previous.norm(dim=-1) > 1e-6)
    draw = torch.rand(len(env_ids), device=self.device)
    kind = torch.full_like(bucket, -1)
    kind[transition] = 2
    kind[transition & eligible & (draw < 0.5)] = 0
    kind[transition & eligible & (draw >= 0.5) & (draw < 0.75)] = 1
    target[kind == 0] = -previous[kind == 0]
    target[kind == 1] = 0
    source = previous
    source[kind == 2] = 0
    abrupt = ~transition | (
      torch.rand(len(env_ids), device=self.device)
      < self.continuation_cfg.abrupt_probability
    )
    self.vel_command_b[env_ids] = torch.where(abrupt[:, None], target, source)
    self.target_command[env_ids] = target
    self.task_class[env_ids] = bucket
    self.transition_kind[env_ids] = kind
    self.transition_age[env_ids] = 0
    self.initialized[env_ids] = True
    self.abrupt[env_ids] = abrupt
    self.sample_counts += torch.bincount(bucket, minlength=7)
    self.is_standing_env[env_ids] = False
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False

  def compute(self, dt, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self._command_dt[ids] = dt
    self.transition_age[ids] += self._command_dt[ids]
    super().compute(dt, env_ids)

  def _update_command(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    current = self.command[ids]
    target = self.target_command[ids]
    delta = target - current
    dt = self._command_dt[ids]
    linear_limit = dt * self.continuation_cfg.linear_acceleration
    factor = (linear_limit / delta[:, :2].norm(dim=-1).clamp(min=1e-12)).clamp(max=1)
    delta[:, :2] *= factor[:, None]
    yaw_limit = dt * self.continuation_cfg.yaw_acceleration
    delta[:, 2] = delta[:, 2].clamp(-yaw_limit, yaw_limit)
    self.vel_command_b[ids] = torch.where(
      self.abrupt[ids, None], target, current + delta
    )
    self._resolve_command(self.robot.data.heading_w, env_ids)

  def _update_metrics(self):
    super()._update_metrics()
    total = self.sample_counts.sum().clamp(min=1)
    for index, name in enumerate(BUCKET_NAMES):
      self._env.extras["log"][f"ContinuationBuckets/{name}"] = (
        (self.task_class == index).float().mean()
      )
      self._env.extras["log"][f"ContinuationSamples/{name}"] = (
        self.sample_counts[index] / total
      )


@dataclass(kw_only=True)
class ContinuationVelocityCommandCfg(OmniVelocityCommandCfg):
  stage: str = "stage1"
  abrupt_probability: float = 0.3
  linear_acceleration: float = 1.0
  yaw_acceleration: float = 0.5

  def build(self, env):
    if self.stage not in STAGE_WEIGHTS:
      raise ValueError("stage must be stage1 or stage2")
    if not 0 <= self.abrupt_probability <= 1:
      raise ValueError("abrupt_probability must be in [0, 1]")
    if self.linear_acceleration <= 0 or self.yaw_acceleration <= 0:
      raise ValueError("accelerations must be positive")
    return ContinuationVelocityCommand(self, env)
