# Copyright (c) 2025 Deep Robotics
# Copyright (c) 2024-2025 Ziqi Fan
# Copyright 2025, The mjlab Developers
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
"""Wolf reward kernels; return unweighted per-environment values.

Turning rewards adapt DeepRoboticsLab/rl_training@9c0e710.
Licenses and provenance: THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from alldog_mjlab.robots.wolf.wolf_constants import (
    WOLF_JOINTS_PER_LEG,
    WOLF_LEG_ORDER,
)

from .common import (
    pure_yaw_command,
    standing_command,
    wheel_contact_sensor,
    wheel_slip_velocities,
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
    standing = standing_command(env, command_name, command_threshold, yaw_threshold)
    return error.abs().sum(dim=1) * standing


def stand_still_joint_velocity_l2(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    """Penalize leg cycling at zero command, even near the default pose."""
    standing = standing_command(env, command_name, command_threshold, yaw_threshold)
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
    sensor = wheel_contact_sensor(env, sensor_name)
    standing = standing_command(env, command_name, command_threshold, yaw_threshold)
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
    sensor = wheel_contact_sensor(env, sensor_name)
    contact_force = sensor.data.force
    assert contact_force is not None
    horizontal_force = torch.norm(contact_force[..., :2], dim=-1)
    contact_gate = (horizontal_force >= min_horizontal_force).to(horizontal_force.dtype)
    asset: Entity = env.scene[asset_cfg.name]
    wheel_velocity_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    upward_velocity = torch.clamp(wheel_velocity_w[..., 2], min=0.0)
    return torch.sum(contact_gate * upward_velocity, dim=1)


def wheel_lateral_slip_l2(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize loaded-wheel lateral material-point speed squared, in m²/s²."""
    lateral, _, _ = wheel_slip_velocities(
        env, sensor_name, asset_cfg, contact_force_threshold
    )
    return lateral.square().sum(dim=1)


def soft_landing(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize net wheel force on first contact, including at zero command.

    Uses MjLab 1.6.0 soft_landing's instantaneous force-norm semantics.
    This is not a substep peak-force or impulse measurement.
    """
    sensor = wheel_contact_sensor(env, sensor_name)
    force = sensor.data.force
    assert force is not None
    landed = sensor.compute_first_contact(env.step_dt)
    impact = force.norm(dim=-1) * landed
    env.extras["log"]["Metrics/landing_force_mean"] = (
        impact.sum() / landed.sum().clamp_min(1)
    )
    return impact.sum(dim=1)


# M20-derived yaw rewards; shared contact/command inputs come from common.py.
_DIAGONAL_A = (0, 3)
_DIAGONAL_B = (1, 2)


def turn_hip_default(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    lin_vel_threshold: float,
    ang_vel_threshold: float,
) -> torch.Tensor:
    """Keep hip-roll near default during pure yaw; leave sagittal joints free."""
    data = env.scene[asset_cfg.name].data
    error = data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[
        :, asset_cfg.joint_ids
    ]
    return error.norm(dim=1) * pure_yaw_command(
        env, command_name, lin_vel_threshold, ang_vel_threshold
    )


def feet_slide_ang_z_cmd(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    lin_vel_threshold: float,
    ang_vel_threshold: float,
) -> torch.Tensor:
    """Upstream wheel-body motion cost, distinct from contact-point slip.

    Subtract root translation, express in the body frame, and penalize both
    planar components while a wheel touches the ground. Wheel spin is not
    subtracted: this term deliberately favors unloading wheels during yaw.
    """
    sensor = wheel_contact_sensor(env, sensor_name)
    # Upstream uses a >1 N net-force history gate; here the native maxforce
    # representative provides the current contact force for each wheel.
    contacts = sensor.data.force.norm(dim=-1) > 1.0
    data = env.scene[asset_cfg.name].data
    velocity = data.body_link_lin_vel_w[:, asset_cfg.body_ids] - (
        data.root_link_lin_vel_w[:, None, :]
    )
    quat = data.root_link_quat_w[:, None, :].expand(-1, velocity.shape[1], -1)
    velocity_b = quat_apply_inverse(quat, velocity)
    cost = (velocity_b[..., :2].norm(dim=-1) * contacts).sum(dim=1)
    return cost * pure_yaw_command(env, command_name, lin_vel_threshold, ang_vel_threshold)


class FeetAirTimeYaw(ManagerTermBase):
    """Reward yaw air time on landing, with clearance since the last contact."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._ground_height = torch.zeros(self.num_envs, 4, device=self.device)
        self._max_air_height = torch.zeros_like(self._ground_height)
        self._initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._ground_height[ids] = 0.0
        self._max_air_height[ids] = 0.0
        self._initialized[ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        asset_cfg: SceneEntityCfg,
        threshold: float,
        cmd_threshold: float,
        foot_height_threshold: float,
    ) -> torch.Tensor:
        sensor = wheel_contact_sensor(env, sensor_name)
        data = env.scene[asset_cfg.name].data
        height = data.body_link_pos_w[:, asset_cfg.body_ids, 2]
        initialized = self._initialized[:, None]
        ground = torch.where(initialized, self._ground_height, height)
        maximum = torch.where(initialized, self._max_air_height, height)
        in_contact = sensor.data.current_contact_time > 0.0
        maximum = torch.where(in_contact, maximum, torch.maximum(maximum, height))
        cleared = maximum - ground > foot_height_threshold
        reward = (
            (sensor.data.last_air_time - threshold)
            * sensor.compute_first_contact(env.step_dt)
            * cleared
        ).sum(dim=1)
        self._ground_height.copy_(torch.where(in_contact, height, ground))
        self._max_air_height.copy_(torch.where(in_contact, height, maximum))
        self._initialized.fill_(True)
        command = env.command_manager.get_command(command_name)
        return reward * (command[:, 2].abs() > cmd_threshold)


def rotation_gait_status(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    lin_vel_threshold: float,
    ang_vel_threshold: float,
) -> torch.Tensor:
    """One diagonal supports while the other is higher, as in upstream M20."""
    sensor = wheel_contact_sensor(env, sensor_name)
    contact = sensor.data.current_contact_time > 0.0
    z = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    difference = z[:, _DIAGONAL_B].mean(1) - z[:, _DIAGONAL_A].mean(1)
    a_supports = contact[:, _DIAGONAL_A].all(1) & (difference > target_height)
    b_supports = contact[:, _DIAGONAL_B].all(1) & (-difference > target_height)
    return (a_supports | b_supports).float() * pure_yaw_command(
        env, command_name, lin_vel_threshold, ang_vel_threshold
    )


class RotationGaitSymmetry(ManagerTermBase):
    """Match each diagonal's contact duty; this is not a policy symmetry loss.

    A running sum avoids reducing the full window each step. Counts and warmup
    are per environment, including after partial resets. Empty buffer slots do
    not count as airborne samples.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._window = max(1, round(cfg.params["window_s"] / env.step_dt))
        self._buffer = torch.zeros(self.num_envs, 2, self._window, device=self.device)
        self._sum = torch.zeros(self.num_envs, 2, device=self.device)
        self._count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._index = 0

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._buffer[ids] = 0.0
        self._sum[ids] = 0.0
        self._count[ids] = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        target_duty: float,
        std: float,
        window_s: float,
        lin_vel_threshold: float,
        ang_vel_threshold: float,
    ) -> torch.Tensor:
        sensor = wheel_contact_sensor(env, sensor_name)
        contact = (sensor.data.current_contact_time > 0.0).float()
        groups = torch.stack(
            (contact[:, _DIAGONAL_A].mean(1), contact[:, _DIAGONAL_B].mean(1)), dim=1
        )
        self._sum.add_(groups - self._buffer[:, :, self._index])
        self._buffer[:, :, self._index] = groups
        self._index = (self._index + 1) % self._window
        self._count.add_(1).clamp_(max=self._window)
        duty = self._sum / self._count[:, None]
        reward = torch.exp(-((duty - target_duty) / std).square().sum(dim=1))
        reward *= self._count / self._window
        return reward * pure_yaw_command(
            env, command_name, lin_vel_threshold, ang_vel_threshold
        )
