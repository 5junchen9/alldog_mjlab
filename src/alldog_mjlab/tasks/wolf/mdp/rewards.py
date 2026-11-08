"""Wolf custom reward kernels; return unweighted per-environment values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from alldog_mjlab.robots.wolf.wolf_constants import (
    WOLF_JOINTS_PER_LEG,
    WOLF_LEG_ORDER,
    WOLF_WHEEL_COLLISION_GEOM_NAMES,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_BASE_HEIGHT_FOOTPRINT_X = 0.3
_BASE_HEIGHT_FOOTPRINT_Y = 0.2
_BASE_HEIGHT_FOOTPRINT_NUM_RAYS = 35
_BASE_HEIGHT_FOOTPRINT_TOLERANCE = 0.0001


def track_linear_velocity_x(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    error = torch.square(command[:, 0] - actual[:, 0])
    return torch.exp(-error / sigma)


def track_linear_velocity_y(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    error = torch.square(command[:, 1] - actual[:, 1])
    return torch.exp(-error / sigma)


def track_angular_velocity_z(
    env: ManagerBasedRlEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    error = torch.square(command[:, 2] - actual[:, 2])
    return torch.exp(-error / sigma)


def vertical_linear_velocity_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def angular_velocity_xy_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    angular_velocity_xy = asset.data.root_link_ang_vel_b[:, :2]
    return torch.sum(torch.square(angular_velocity_xy), dim=1)


def base_height_l2_flat(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Absolute world-height error; retain the existing callable name."""
    asset: Entity = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    return torch.abs(base_height - target_height)


class base_height_l2_terrain(ManagerTermBase):
    """Absolute height error using the mean clearance of the central 35 terrain rays."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        sensor_name: str = cfg.params["sensor_name"]
        pattern = env.scene[sensor_name].cfg.pattern
        offsets, _ = pattern.generate_rays(None, str(self.device))
        tolerance = _BASE_HEIGHT_FOOTPRINT_TOLERANCE
        mask = (offsets[:, 0].abs() <= _BASE_HEIGHT_FOOTPRINT_X + tolerance) & (
            offsets[:, 1].abs() <= _BASE_HEIGHT_FOOTPRINT_Y + tolerance
        )
        self._indices = mask.nonzero(as_tuple=False).squeeze(-1)
        selected = offsets[self._indices]
        assert selected.shape[0] == _BASE_HEIGHT_FOOTPRINT_NUM_RAYS, selected.shape
        assert (
            abs(float(selected[:, 0].abs().max()) - _BASE_HEIGHT_FOOTPRINT_X)
            <= _BASE_HEIGHT_FOOTPRINT_TOLERANCE
        ), float(selected[:, 0].abs().max())
        assert (
            abs(float(selected[:, 1].abs().max()) - _BASE_HEIGHT_FOOTPRINT_Y)
            <= _BASE_HEIGHT_FOOTPRINT_TOLERANCE
        ), float(selected[:, 1].abs().max())

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        target_height: float,
        sensor_name: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        raw_scan = envs_mdp.height_scan(env, sensor_name=sensor_name)
        base_height = raw_scan[:, self._indices].mean(dim=1)
        return torch.abs(base_height - target_height)


def all_position_default(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sum squared leg-position errors from default, for every command mode."""
    data = env.scene[asset_cfg.name].data
    default_joint_pos = data.default_joint_pos
    assert default_joint_pos is not None
    error = data.joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[
        :, asset_cfg.joint_ids
    ]
    return error.square().sum(dim=1)


def stand_still(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    command = env.command_manager.get_command(command_name)
    standing = (command[:, :2].norm(dim=1) < command_threshold) & (
        command[:, 2].abs() < yaw_threshold
    )
    return error.abs().sum(dim=1) * standing


def stand_still_joint_velocity_l2(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    """Penalize leg cycling at zero command, even near the default pose."""
    command = env.command_manager.get_command(command_name)
    standing = (command[:, :2].norm(dim=1) < command_threshold) & (
        command[:, 2].abs() < yaw_threshold
    )
    velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return velocity.square().sum(dim=1) * standing


def stand_still_wheel_air(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    command_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    """Favor four-wheel support when no planar motion is requested."""
    sensor = env.scene[sensor_name]
    assert tuple(sensor.primary_names) == WOLF_WHEEL_COLLISION_GEOM_NAMES
    command = env.command_manager.get_command(command_name)
    standing = (command[:, :2].norm(dim=1) < command_threshold) & (
        command[:, 2].abs() < yaw_threshold
    )
    return (sensor.data.current_contact_time <= 0).sum(dim=1) * standing


# Policy action channels are interleaved as hip/thigh/calf/wheel for each leg.
_BLOCK_DIM = len(WOLF_JOINTS_PER_LEG)
_LEG_CHANNELS = tuple(
    leg * _BLOCK_DIM + joint
    for leg in range(len(WOLF_LEG_ORDER))
    for joint in range(_BLOCK_DIM - 1)
)
_WHEEL_CHANNELS = tuple(
    leg * _BLOCK_DIM + (_BLOCK_DIM - 1) for leg in range(len(WOLF_LEG_ORDER))
)


def leg_action_rate_l2(env) -> torch.Tensor:
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    diff = action[:, list(_LEG_CHANNELS)] - prev_action[:, list(_LEG_CHANNELS)]
    return torch.sum(torch.square(diff), dim=1)


def wheel_action_rate_l2(env) -> torch.Tensor:
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    diff = action[:, list(_WHEEL_CHANNELS)] - prev_action[:, list(_WHEEL_CHANNELS)]
    return torch.sum(torch.square(diff), dim=1)


def orientation_l1(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.projected_gravity_b[:, :2]), dim=1)


def hip_default_l1(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    y_ref: float,
    yaw_ref: float,
    y_scale: float,
    yaw_scale: float,
    min_scale: float,
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    hip_error = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    command = env.command_manager.get_command(command_name)
    alpha = 1.0
    alpha = alpha - y_scale * torch.clamp(
        torch.abs(command[:, 1]) / max(y_ref, 1e-06), max=1.0
    )
    alpha = alpha - yaw_scale * torch.clamp(
        torch.abs(command[:, 2]) / max(yaw_ref, 1e-06), max=1.0
    )
    alpha = torch.clamp(alpha, min=min_scale, max=1.0)
    return hip_error * alpha


def run_still_leg_l1(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    x_threshold: float,
    y_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    command = env.command_manager.get_command(command_name)
    active = (
        (torch.abs(command[:, 0]) > x_threshold)
        & (torch.abs(command[:, 1]) < y_threshold)
        & (torch.abs(command[:, 2]) < yaw_threshold)
    ).to(torch.float32)
    leg_error = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return leg_error * active


def wheel_force_lift(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    min_horizontal_force: float,
) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    assert list(sensor.primary_names) == list(WOLF_WHEEL_COLLISION_GEOM_NAMES), (
        f"wheel contact sensor primary order {sensor.primary_names} != policy wheel order {WOLF_WHEEL_COLLISION_GEOM_NAMES}"
    )
    contact_force = sensor.data.force
    assert contact_force is not None
    horizontal_force = torch.norm(contact_force[..., :2], dim=-1)
    contact_gate = (horizontal_force >= min_horizontal_force).to(horizontal_force.dtype)
    asset: Entity = env.scene[asset_cfg.name]
    wheel_velocity_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    upward_velocity = torch.clamp(wheel_velocity_w[..., 2], min=0.0)
    return torch.sum(contact_gate * upward_velocity, dim=1)
