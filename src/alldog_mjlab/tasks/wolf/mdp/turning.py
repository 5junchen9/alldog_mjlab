# Copyright (c) 2025 Deep Robotics
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
"""M20-derived turn rewards bounded and conditioned on Wolf's yaw progress.

Source: DeepRoboticsLab/rl_training@9c0e710, velocity/mdp/rewards.py.
Licenses: THIRD_PARTY_NOTICES.md.
Wolf uses per-env reset/mode state and a swing-height cost. The upstream
global terrain scalar is omitted so flat training receives these terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from alldog_mjlab.robots.wolf.wolf_constants import WOLF_WHEEL_COLLISION_GEOM_NAMES

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# All sensor/body selectors use the explicit FL, FR, RL, RR policy order.
_DIAGONAL_A = (0, 3)
_DIAGONAL_B = (1, 2)


def _contact_sensor(env: ManagerBasedRlEnv, sensor_name: str):
    sensor = env.scene[sensor_name]
    assert tuple(sensor.primary_names) == WOLF_WHEEL_COLLISION_GEOM_NAMES
    return sensor


def _turn_gate(env, command_name, lin_vel_threshold, ang_vel_threshold):
    command = env.command_manager.get_command(command_name)
    return (command[:, :2].norm(dim=1) < lin_vel_threshold) & (
        command[:, 2].abs() > ang_vel_threshold
    )


def _yaw_tracking_progress(env, command_name):
    """No gait bonus for stationary/wrong-way stepping; cap at full tracking."""
    requested = env.command_manager.get_command(command_name)[:, 2]
    actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return (actual * requested.sign() / requested.abs().clamp_min(1e-6)).clamp(0, 1)


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
    return error.norm(dim=1) * _turn_gate(
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
    sensor = _contact_sensor(env, sensor_name)
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
    return cost * _turn_gate(env, command_name, lin_vel_threshold, ang_vel_threshold)


class FeetAirTimeYaw(ManagerTermBase):
    """Bound yaw air time/clearance and penalize excessive swing height."""

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
        lin_vel_threshold: float,
        max_air_time: float,
        max_foot_height: float,
        height_penalty_scale: float,
    ) -> torch.Tensor:
        sensor = _contact_sensor(env, sensor_name)
        data = env.scene[asset_cfg.name].data
        height = data.body_link_pos_w[:, asset_cfg.body_ids, 2]
        initialized = self._initialized[:, None]
        ground = torch.where(initialized, self._ground_height, height)
        maximum = torch.where(initialized, self._max_air_height, height)
        in_contact = sensor.data.current_contact_time > 0.0
        maximum = torch.where(in_contact, maximum, torch.maximum(maximum, height))
        peak_clearance = maximum - ground
        cleared = (peak_clearance > foot_height_threshold) & (
            peak_clearance <= max_foot_height
        )
        landing_bonus = (
            (sensor.data.last_air_time.clamp(max=max_air_time) - threshold)
            * sensor.compute_first_contact(env.step_dt)
            * cleared
        ).sum(dim=1)
        # A continuous cost is needed: merely removing the landing bonus above
        # a ceiling would still leave high stepping unpenalized during flight.
        clearance = height - ground
        height_cost = (
            (clearance - max_foot_height).clamp_min(0).square() * ~in_contact
        ).sum(dim=1)
        self._ground_height.copy_(torch.where(in_contact, height, ground))
        self._max_air_height.copy_(torch.where(in_contact, height, maximum))
        self._initialized.fill_(True)
        gate = _turn_gate(env, command_name, lin_vel_threshold, cmd_threshold)
        return gate * (
            landing_bonus * _yaw_tracking_progress(env, command_name)
            - height_penalty_scale * height_cost
        )


def rotation_gait_status(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    height_std: float,
    lin_vel_threshold: float,
    ang_vel_threshold: float,
) -> torch.Tensor:
    """Favor a modest diagonal swing height while actually turning."""
    sensor = _contact_sensor(env, sensor_name)
    contact = sensor.data.current_contact_time > 0.0
    z = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    difference = z[:, _DIAGONAL_B].mean(1) - z[:, _DIAGONAL_A].mean(1)
    a_supports = contact[:, _DIAGONAL_A].all(1) & ~contact[:, _DIAGONAL_B].any(1)
    b_supports = contact[:, _DIAGONAL_B].all(1) & ~contact[:, _DIAGONAL_A].any(1)
    a_score = a_supports * torch.exp(
        -((difference - target_height) / height_std).square()
    )
    b_score = b_supports * torch.exp(
        -((-difference - target_height) / height_std).square()
    )
    return (
        torch.maximum(a_score, b_score)
        * _turn_gate(env, command_name, lin_vel_threshold, ang_vel_threshold)
        * _yaw_tracking_progress(env, command_name)
    )


class RotationGaitSymmetry(ManagerTermBase):
    """Match each diagonal's contact duty while tracking a turn command.

    A running sum avoids reducing the full window each step. Counts and warmup
    are per environment, including after partial resets and leaving turn mode.
    Empty buffer slots do not count as airborne samples.
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
        sensor = _contact_sensor(env, sensor_name)
        gate = _turn_gate(env, command_name, lin_vel_threshold, ang_vel_threshold)
        self.reset(~gate)
        contact = (sensor.data.current_contact_time > 0.0).float()
        groups = torch.stack(
            (contact[:, _DIAGONAL_A].mean(1), contact[:, _DIAGONAL_B].mean(1)), dim=1
        ) * gate[:, None]
        self._sum.add_(groups - self._buffer[:, :, self._index])
        self._buffer[:, :, self._index] = groups
        self._index = (self._index + 1) % self._window
        self._count.add_(gate).clamp_(max=self._window)
        duty = self._sum / self._count[:, None].clamp_min(1)
        reward = torch.exp(-((duty - target_duty) / std).square().sum(dim=1))
        reward *= self._count / self._window
        return reward * gate * _yaw_tracking_progress(env, command_name)
