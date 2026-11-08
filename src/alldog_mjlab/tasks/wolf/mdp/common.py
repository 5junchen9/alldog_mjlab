"""Shared Wolf contact data, wheel kinematics and command predicates.

These helpers contain no reward weights, reward accumulation or reward calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from alldog_mjlab.robots.wolf.wolf_constants import (
    WOLF_WHEEL_BODY_NAMES,
    WOLF_WHEEL_COLLISION_GEOM_NAMES,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor


def wheel_contact_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> ContactSensor:
    """Resolve a four-wheel sensor in explicit FL/FR/RL/RR order."""
    sensor = env.scene[sensor_name]
    assert tuple(sensor.primary_names) == WOLF_WHEEL_COLLISION_GEOM_NAMES
    return sensor


def standing_command(env, command_name, command_threshold, yaw_threshold):
    command = env.command_manager.get_command(command_name)
    return (command[:, :2].norm(dim=1) < command_threshold) & (
        command[:, 2].abs() < yaw_threshold
    )


def pure_yaw_command(env, command_name, lin_vel_threshold, ang_vel_threshold):
    command = env.command_manager.get_command(command_name)
    return (command[:, :2].norm(dim=1) < lin_vel_threshold) & (
        command[:, 2].abs() > ang_vel_threshold
    )


def yaw_tracking_progress(env, command_name):
    """Fraction of requested yaw achieved in the requested direction, capped at 1."""
    requested = env.command_manager.get_command(command_name)[:, 2]
    actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return (actual * requested.sign() / requested.abs().clamp_min(1e-6)).clamp(0, 1)


def wheel_slip_velocities(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed lateral/longitudinal slip [B,4] and a loaded-contact mask.

    The native maxforce sensor supplies one representative contact per wheel,
    with force in its contact frame and position/normal in world coordinates.
    Evaluate the wheel material point velocity v_O + omega x (p_contact-p_O),
    including the wheel body's total angular velocity, against stationary terrain.
    The wheel joint/cylinder axis is local Z in Wolf's MJCF. Its cross product
    with the contact normal defines rolling direction; signs follow these axes,
    rather than the policy's forward wheel-velocity convention.

    Airborne/unloaded wheels and contacts with no defined rolling direction are
    zeroed. Units are m/s. Multiple contacts are represented by the strongest one.
    """
    sensor = wheel_contact_sensor(env, sensor_name)
    assert tuple(asset_cfg.body_names) == WOLF_WHEEL_BODY_NAMES
    assert sensor.cfg.reduce == "maxforce" and not sensor.cfg.global_frame
    contact = sensor.data
    assert contact.found is not None and contact.force is not None
    assert contact.pos is not None and contact.normal is not None
    robot = env.scene[asset_cfg.name]
    positions = robot.data.body_link_pos_w[:, asset_cfg.body_ids]
    velocities = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    angular = robot.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
    quaternions = robot.data.body_link_quat_w[:, asset_cfg.body_ids]

    local_axis = torch.zeros_like(positions)
    local_axis[..., 2] = 1.0
    axle = quat_apply(quaternions, local_axis)
    normal = torch.nn.functional.normalize(contact.normal, dim=-1)
    rolling = torch.linalg.cross(axle, normal, dim=-1)
    rolling_norm = rolling.norm(dim=-1)
    rolling = rolling / rolling_norm.clamp_min(1e-6).unsqueeze(-1)
    lateral = torch.linalg.cross(normal, rolling, dim=-1)
    loaded = (
        (contact.found > 0)
        & (contact.force[..., 0].abs() > contact_force_threshold)
        & (rolling_norm > 1e-6)
    )
    point_velocity = velocities + torch.linalg.cross(
        angular, contact.pos - positions, dim=-1
    )
    lateral_speed = (point_velocity * lateral).sum(dim=-1)
    longitudinal_speed = (point_velocity * rolling).sum(dim=-1)
    return (
        torch.where(loaded, lateral_speed, 0.0),
        torch.where(loaded, longitudinal_speed, 0.0),
        loaded,
    )
