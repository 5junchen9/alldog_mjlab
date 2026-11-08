"""Uniform Wolf velocity samples with a Go2W-style small-XY deadband."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class WolfVelocityCommand(UniformVelocityCommand):
    """Use native sampling for every environment; clear small sampled XY only."""

    cfg: WolfVelocityCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        moving = (
            self.vel_command_b[env_ids, :2].norm(dim=1) > self.cfg.small_xy_threshold
        )
        self.vel_command_b[env_ids, :2] *= moving.unsqueeze(1)


@dataclass(kw_only=True)
class WolfVelocityCommandCfg(UniformVelocityCommandCfg):
    small_xy_threshold: float = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.small_xy_threshold < 0.0:
            raise ValueError("small_xy_threshold must be nonnegative")

    def build(self, env: ManagerBasedRlEnv) -> WolfVelocityCommand:
        return WolfVelocityCommand(self, env)
