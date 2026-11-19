"""Original flat command cycles; independently sampled rough motion envelopes."""

from dataclasses import dataclass

import torch

from mjlab.tasks.velocity.config.black_teacher.teacherfree_command import (
  TeacherFreeVelocityCommand,
  TeacherFreeVelocityCommandCfg,
  allocate_profiles,
)

from .terrain_cfgs import domain_counts


def rough_targets(level: torch.Tensor):
  count, device = len(level), level.device
  limits = torch.tensor(
    [[0.30, 0.20, 0.25], [0.50, 0.30, 0.35], [0.70, 0.40, 0.40], [0.70, 0.40, 0.40]],
    device=device,
  )[level.clamp(1, 4) - 1]
  draw = torch.rand(count, device=device)
  # 40% forward/back, 20% lateral, 20% yaw, 15% mixed, 5% stand.
  axis = torch.where(draw < 0.4, 0, torch.where(draw < 0.6, 1, 2))
  target = torch.zeros(count, 3, device=device)
  sign = torch.randint(2, (count,), device=device) * 2 - 1
  target[torch.arange(count, device=device), axis] = sign * (
    0.05
    + torch.rand(count, device=device)
    * (limits[torch.arange(count, device=device), axis] - 0.05)
  )
  mixed = (draw >= 0.8) & (draw < 0.95)
  angle = torch.rand(count, device=device) * 2 * torch.pi
  speed = 0.05 + torch.rand(count, device=device) * (limits[:, 0] - 0.05)
  target[mixed, 0] = (speed * angle.cos())[mixed]
  target[mixed, 1] = (speed * angle.sin())[mixed].clamp(
    -limits[mixed, 1], limits[mixed, 1]
  )
  target[mixed, 2] = ((torch.rand(count, device=device) * 2 - 1) * limits[:, 2])[mixed]
  target[draw >= 0.95] = 0
  return target, torch.where(draw >= 0.95, 4, torch.where(mixed, 3, axis))


class RoughVelocityCommand(TeacherFreeVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    counts = domain_counts(self.num_envs)
    self.flat_mask = torch.arange(self.num_envs, device=self.device) < counts[0]
    self.profile_id[: counts[0]] = allocate_profiles(
      counts[0], self.device, cfg.profile_seed
    )
    self.rough_motion = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    terrain = env.scene.terrain
    terrain.terrain_types[:] = torch.repeat_interleave(
      torch.arange(7, device=self.device), torch.tensor(counts, device=self.device)
    )
    terrain.terrain_levels[:] = (~self.flat_mask).long()
    terrain.env_origins[:] = terrain.terrain_origins[
      terrain.terrain_levels, terrain.terrain_types
    ]
    self.cross_seconds = torch.zeros(7, 5, 9, 4, 2, device=self.device)

  def _resample_command(self, env_ids):
    flat = env_ids[self.flat_mask[env_ids]]
    if len(flat):
      super()._resample_command(flat)
    rough = env_ids[~self.flat_mask[env_ids]]
    if not len(rough):
      return
    terrain = self._env.scene.terrain
    assert terrain is not None
    levels = terrain.terrain_levels[rough]
    target, motion = rough_targets(levels)
    self.rough_motion[rough] = motion
    self.target_command[rough] = target
    self.cycle_phase[rough] = -1
    self.task_class[rough] = torch.where(motion == 4, 6, torch.where(motion == 3, 1, 0))
    probability = torch.where(levels >= 3, 0.5, torch.where(levels >= 2, 0.25, 0.0))
    self.abrupt[rough] = torch.rand(len(rough), device=self.device) < probability
    self.time_left[rough] = torch.empty(len(rough), device=self.device).uniform_(3, 8)
    self.initialized[rough] = True
    self.is_standing_env[rough] = False
    self.is_heading_env[rough] = False
    self.is_world_env[rough] = False
    self.is_forward_env[rough] = False
    immediate = rough[self.abrupt[rough]]
    self.vel_command_b[immediate] = self.target_command[immediate]

  def _update_command(self, env_ids=None):
    super()._update_command(env_ids)
    if not hasattr(self, "cross_seconds"):
      return
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    command = self.command[ids]
    axis = command.abs().argmax(-1)
    direction = axis * 2 + (command.gather(1, axis[:, None]).squeeze(1) < 0).long()
    mixed = (command.abs() > 0.04).sum(-1) > 1
    direction = torch.where(mixed, 6 + (command[:, 2] < 0).long(), direction)
    speed = torch.maximum(command[:, :2].norm(dim=-1), command[:, 2].abs())
    direction = torch.where(speed < 0.04, 8, direction)
    speed_bin = torch.bucketize(
      speed, torch.tensor([0.04, 0.15, 0.5], device=self.device)
    )
    terrain = self._env.scene.terrain
    assert terrain is not None
    index = (
      ((terrain.terrain_types[ids] * 5 + terrain.terrain_levels[ids]) * 9 + direction)
      * 4
      + speed_bin
    ) * 2 + self.abrupt[ids].long()
    self.cross_seconds.view(-1).index_add_(0, index, self._command_dt[ids])

  def _update_metrics(self):
    super()._update_metrics()
    terrain = self._env.scene.terrain
    assert terrain is not None
    for kind in range(7):
      mask = terrain.terrain_types == kind
      self._env.extras["log"][f"Rough/domain_{kind}_fraction"] = mask.float().mean()
    self._env.extras["log"]["Rough/flat_fraction"] = self.flat_mask.float().mean()


@dataclass(kw_only=True)
class RoughVelocityCommandCfg(TeacherFreeVelocityCommandCfg):
  def build(self, env):
    return RoughVelocityCommand(self, env)
