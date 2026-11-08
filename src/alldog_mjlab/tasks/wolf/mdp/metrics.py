"""Raw Wolf slip diagnostics, independent of reward registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .common import wheel_slip_velocities

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def wheel_lateral_slip_l2(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Sum loaded-wheel lateral slip squared, in m²/s², without dt scaling."""
    lateral, _, _ = wheel_slip_velocities(
        env, sensor_name, asset_cfg, contact_force_threshold
    )
    return lateral.square().sum(dim=1)


def wheel_longitudinal_slip_l2(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Sum loaded-wheel longitudinal slip squared, in m²/s², without dt scaling."""
    _, longitudinal, _ = wheel_slip_velocities(
        env, sensor_name, asset_cfg, contact_force_threshold
    )
    return longitudinal.square().sum(dim=1)
