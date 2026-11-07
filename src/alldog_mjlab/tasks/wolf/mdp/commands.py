"""M20-style special command samples on MjLab's native velocity command.

Sampling design: DeepRoboticsLab/rl_training@9c0e710, velocity/mdp/commands.py.
"""

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
    """Reserve disjoint standing / lateral / longitudinal / yaw samples.

    DeepRoboticsLab/rl_training, commit 9c0e710, uses these special samples
    outside its command curriculum. Wolf keeps direct body-frame yaw velocity
    commands rather than introducing a heading target.
    """

    cfg: WolfVelocityCommandCfg

    def __init__(self, cfg: WolfVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._special_ranges = (
            tuple(cfg.ranges.lin_vel_x),
            tuple(cfg.ranges.lin_vel_y),
            tuple(cfg.ranges.ang_vel_z),
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        slot = torch.rand(len(env_ids), device=self.device)
        boundary = self.cfg.rel_standing_envs
        standing = slot < boundary
        self.is_standing_env[env_ids] = standing
        self.vel_command_b[env_ids[standing]] = 0.0

        # Native heading/world/forward modes apply only to mixed samples.
        special = standing.clone()
        for axis, fraction in (
            (1, self.cfg.rel_pure_lateral_envs),
            (0, self.cfg.rel_pure_longitudinal_envs),
            (2, self.cfg.rel_pure_yaw_envs),
        ):
            selected = (slot >= boundary) & (slot < boundary + fraction)
            ids = env_ids[selected]
            self.vel_command_b[ids] = 0.0
            self.vel_command_b[ids, axis] = torch.empty(
                len(ids), device=self.device
            ).uniform_(*self._special_ranges[axis])
            special |= selected
            boundary += fraction
        special_ids = env_ids[special]
        self.is_heading_env[special_ids] = False
        self.is_world_env[special_ids] = False
        self.is_forward_env[special_ids] = False


@dataclass(kw_only=True)
class WolfVelocityCommandCfg(UniformVelocityCommandCfg):
    rel_pure_lateral_envs: float = 0.02
    rel_pure_longitudinal_envs: float = 0.02
    rel_pure_yaw_envs: float = 0.20

    def __post_init__(self) -> None:
        super().__post_init__()
        fractions = (
            self.rel_standing_envs,
            self.rel_pure_lateral_envs,
            self.rel_pure_longitudinal_envs,
            self.rel_pure_yaw_envs,
        )
        if any(not 0.0 <= p <= 1.0 for p in fractions) or sum(fractions) > 1.0:
            raise ValueError("Wolf special command fractions must sum to at most 1")
        if self.rel_forward_envs != 0.0:
            raise ValueError(
                "Use rel_pure_longitudinal_envs for Wolf straight commands"
            )

    def build(self, env: ManagerBasedRlEnv) -> WolfVelocityCommand:
        return WolfVelocityCommand(self, env)
