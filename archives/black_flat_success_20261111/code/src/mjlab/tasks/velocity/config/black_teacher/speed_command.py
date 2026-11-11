"""Explicit speed envelopes for independent Black training projects."""

import math
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

from .omni_command import OmniVelocityCommand, OmniVelocityCommandCfg


class ConfiguredOmniVelocityCommand(OmniVelocityCommand):
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Preserve the uniform mixed-command coverage without Omni's 0.1 clamp.
    UniformVelocityCommand._resample_command(self, env_ids)
    probabilities = torch.tensor(
      [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.15, 0.05], device=self.device
    )
    category = torch.multinomial(probabilities, len(env_ids), replacement=True)
    commands = self.vel_command_b[env_ids].clone()
    assert isinstance(self.cfg, ConfiguredOmniVelocityCommandCfg)
    minimum = self.cfg.min_speeds
    assert minimum is not None
    maximum = (
      self.cfg.ranges.lin_vel_x[1],
      self.cfg.ranges.lin_vel_y[1],
      self.cfg.ranges.ang_vel_z[1],
    )
    for index, axis, sign in (
      (0, 0, 1),
      (1, 0, -1),
      (2, 2, 1),
      (3, 2, -1),
      (4, 1, 1),
      (5, 1, -1),
    ):
      selected = category == index
      speed = torch.empty(len(env_ids), device=self.device).uniform_(
        minimum[axis], maximum[axis]
      )
      commands[selected] = 0.0
      commands[selected, axis] = sign * speed[selected]
    commands[category == 7] = 0.0
    self.vel_command_b[env_ids] = commands
    self.is_standing_env[env_ids] = category == 7
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False


@dataclass(kw_only=True)
class ConfiguredOmniVelocityCommandCfg(OmniVelocityCommandCfg):
  min_speeds: tuple[float, float, float] | None = None
  """Required positive minimum magnitudes for pure vx, vy and yaw commands.

  No low/high speed preset is approved by default. Mixed commands retain
  uniform symmetric coverage, including values below these pure-axis minima.
  """

  def validate_speeds(self) -> None:
    if self.min_speeds is None or len(self.min_speeds) != 3:
      raise ValueError("Set explicit min_speeds and symmetric maximum speed ranges")
    ranges = (self.ranges.lin_vel_x, self.ranges.lin_vel_y, self.ranges.ang_vel_z)
    for name, minimum, bounds in zip(
      ("vx", "vy", "yaw"), self.min_speeds, ranges, strict=True
    ):
      if len(bounds) != 2 or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in (minimum, *bounds)
      ):
        raise ValueError(f"{name} speed bounds must be finite numbers")
      lower, upper = bounds
      if lower != -upper or not 0 < minimum <= upper:
        raise ValueError(f"{name} requires symmetric ranges and 0 < min <= max")

  def build(self, env: ManagerBasedRlEnv) -> ConfiguredOmniVelocityCommand:
    self.validate_speeds()
    return ConfiguredOmniVelocityCommand(self, env)
