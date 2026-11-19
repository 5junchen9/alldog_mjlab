"""Exact formal composition, explicitly scaled diagnostic sample domains."""

from dataclasses import dataclass

import torch

from mjlab.tasks.velocity.config.black_rough.command import (
  RoughVelocityCommand,
  RoughVelocityCommandCfg,
)
from mjlab.tasks.velocity.config.black_rough.terrain_cfgs import (
  ROUGH_WEIGHTS,
  counts_for,
)
from mjlab.tasks.velocity.config.black_teacher.teacherfree_command import (
  TeacherFreeVelocityCommand,
  allocate_profiles,
)
from support.black.typed_runtime import require_terrain


class CompositionCommand(RoughVelocityCommand):
  cfg: "CompositionCommandCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    n = self.num_envs
    if n < 28 or n % 4:
      raise ValueError("Diagnostic sample must be >=28 and divisible by four")
    flat = n // 4
    self.flat_mask[:] = torch.arange(n, device=self.device) < flat
    counts = [flat] + counts_for(n - flat, ROUGH_WEIGHTS)
    counts[-1] += n - sum(counts)
    terrain = require_terrain(env)
    terrain.terrain_types[:] = torch.repeat_interleave(
      torch.arange(7, device=self.device), torch.tensor(counts, device=self.device)
    )
    terrain.terrain_levels[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[
      terrain.terrain_levels, terrain.terrain_types
    ]
    self.profile_id[:flat] = allocate_profiles(flat, self.device, cfg.profile_seed)
    self.composition_profile_seed = cfg.profile_seed
    self.cross_seconds = torch.zeros(7, cfg.height_rows, 9, 4, 2, device=self.device)

  def _update_command(self, env_ids=None):
    TeacherFreeVelocityCommand._update_command(self, env_ids)
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
    bins = torch.bucketize(speed, torch.tensor([0.04, 0.15, 0.5], device=self.device))
    terrain = require_terrain(self._env)
    index = (
      (
        (
          terrain.terrain_types[ids] * self.cfg.height_rows
          + terrain.terrain_levels[ids]
        )
        * 9
        + direction
      )
      * 4
      + bins
    ) * 2 + self.abrupt[ids].long()
    self.cross_seconds.view(-1).index_add_(0, index, self._command_dt[ids])

  def composition_report(self):
    return dict(
      num_envs=self.num_envs,
      diagnostic_sample=self.num_envs != 4096,
      domain_counts=torch.bincount(
        require_terrain(self._env).terrain_types, minlength=7
      )
      .cpu()
      .tolist(),
      flat_count=int(self.flat_mask.sum()),
      flat_profile_counts=torch.bincount(self.profile_id[self.flat_mask], minlength=3)
      .cpu()
      .tolist(),
    )


@dataclass(kw_only=True)
class CompositionCommandCfg(RoughVelocityCommandCfg):
  height_rows: int = 5

  def build(self, env):
    return CompositionCommand(self, env)
