"""Explicit first-round low/high sampling candidates, isolated from Omni."""

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv

from .omni_command import OmniVelocityCommand, OmniVelocityCommandCfg


def low_magnitudes(count: int, device: str) -> torch.Tensor:
  band = torch.multinomial(torch.tensor([0.4, 0.4, 0.2], device=device), count, True)
  lo = torch.tensor([0.05, 0.15, 0.30], device=device)[band]
  hi = torch.tensor([0.15, 0.30, 0.50], device=device)[band]
  return lo + torch.rand(count, device=device) * (hi - lo)


def sample_targets(profile: str, count: int, device: str):
  probabilities = (
    [0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]
    if profile == "low"
    else [0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.25, 0.05]
  )
  category = torch.multinomial(torch.tensor(probabilities, device=device), count, True)
  commands = torch.zeros(count, 3, device=device)
  for index, axis, sign in (
    (0, 0, 1),
    (1, 0, -1),
    (2, 2, 1),
    (3, 2, -1),
    (4, 1, 1),
    (5, 1, -1),
  ):
    selected = category == index
    if profile == "low":
      speed = low_magnitudes(count, device)
    else:
      lo, hi = (0.3, 0.5) if axis == 2 else (0.5, 1.0)
      speed = torch.empty(count, device=device).uniform_(lo, hi)
    commands[selected, axis] = sign * speed[selected]
  mixed = category == 6
  if profile == "low":
    # Bounded planar magnitude rather than independent components above 0.5.
    angle = torch.rand(count, device=device) * (2.0 * torch.pi)
    speed = low_magnitudes(count, device)
    commands[mixed, 0] = (speed * angle.cos())[mixed]
    commands[mixed, 1] = (speed * angle.sin())[mixed]
    commands[mixed, 2] = torch.empty(count, device=device).uniform_(-0.3, 0.3)[mixed]
  else:
    signs = torch.randint(0, 2, (count, 3), device=device) * 2 - 1
    magnitudes = torch.rand(count, 3, device=device)
    magnitudes[:, :2] = 0.5 + 0.5 * magnitudes[:, :2]
    magnitudes[:, 2] = 0.3 + 0.2 * magnitudes[:, 2]
    # Half of mixed targets are exact existing per-axis boundaries.
    boundary = torch.rand(count, device=device) < 0.5
    magnitudes[boundary] = torch.tensor([1.0, 1.0, 0.5], device=device)
    commands[mixed] = (signs * magnitudes)[mixed]
  return commands, category


def ramp_command(current, target, dt, linear_acceleration=1.0, yaw_acceleration=0.5):
  delta = target - current
  limit = linear_acceleration * dt
  xy = delta[:, :2]
  xy = xy * (limit / xy.norm(dim=-1).clamp(min=1e-12)).clamp(max=1.0).unsqueeze(-1)
  yaw = delta[:, 2:3].clamp(-yaw_acceleration * dt, yaw_acceleration * dt)
  return current + torch.cat((xy, yaw), dim=-1)


class CandidateVelocityCommand(OmniVelocityCommand):
  def __init__(self, cfg: "CandidateVelocityCommandCfg", env):
    super().__init__(cfg, env)
    self.candidate_cfg = cfg
    self.target_command = torch.zeros_like(self.vel_command_b)
    self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.abrupt = torch.zeros_like(self.initialized)
    self.reversal = torch.zeros_like(self.initialized)
    self.category = torch.full((self.num_envs,), 7, device=self.device)

  def reset(self, env_ids):
    self.initialized[env_ids] = False
    return super().reset(env_ids)

  def _resample_command(self, env_ids):
    target, category = sample_targets(
      self.candidate_cfg.profile, len(env_ids), self.device
    )
    initial = ~self.initialized[env_ids]
    reversal = torch.zeros_like(initial)
    if self.candidate_cfg.profile == "high":
      previous = self.target_command[env_ids]
      reversal = (
        (~initial)
        & (category != 7)
        & (previous.norm(dim=-1) > 0)
        & (
          torch.rand(len(env_ids), device=self.device)
          < self.candidate_cfg.reversal_probability
        )
      )
      target[reversal] = -previous[reversal]
      previous_category = self.category[env_ids]
      opposite = torch.tensor([1, 0, 3, 2, 5, 4, 6, 7], device=self.device)
      category[reversal] = opposite[previous_category[reversal]]
    abrupt = initial | (
      torch.rand(len(env_ids), device=self.device)
      < self.candidate_cfg.abrupt_probability
    )
    if self.candidate_cfg.profile == "low":
      abrupt[:] = True
    self.target_command[env_ids] = target
    self.vel_command_b[env_ids] = torch.where(
      abrupt.unsqueeze(-1), target, self.vel_command_b[env_ids]
    )
    self.initialized[env_ids] = True
    self.abrupt[env_ids] = abrupt
    self.reversal[env_ids] = reversal
    self.category[env_ids] = category
    # A smooth stop must reach zero without Uniform's immediate standing mask.
    self.is_standing_env[env_ids] = False
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False

  def _update_command(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    if self.candidate_cfg.profile == "high":
      current, target = self.vel_command_b[ids], self.target_command[ids]
      updated = ramp_command(
        current,
        target,
        self._env.step_dt,
        self.candidate_cfg.linear_acceleration,
        self.candidate_cfg.yaw_acceleration,
      )
      self.vel_command_b[ids] = torch.where(self.abrupt[ids, None], target, updated)
    self._resolve_command(self.robot.data.heading_w, env_ids)

  def _update_metrics(self):
    super()._update_metrics()
    for name, values in (
      ("target_standing", self.category == 7),
      ("abrupt", self.abrupt),
      ("reversal", self.reversal),
    ):
      self._env.extras["log"][f"CommandTransitions/{name}"] = values.float().mean()


@dataclass(kw_only=True)
class CandidateVelocityCommandCfg(OmniVelocityCommandCfg):
  profile: str
  reversal_probability: float = 0.5
  abrupt_probability: float = 0.3
  linear_acceleration: float = 1.0
  yaw_acceleration: float = 0.5

  def build(self, env: ManagerBasedRlEnv) -> CandidateVelocityCommand:
    if self.profile not in ("low", "high"):
      raise ValueError("Unknown candidate command profile")
    return CandidateVelocityCommand(self, env)
