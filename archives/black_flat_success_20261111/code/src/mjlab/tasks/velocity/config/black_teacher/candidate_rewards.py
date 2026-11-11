"""Isolated candidate rewards; original Omni implementations stay unchanged."""

from __future__ import annotations

import math

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import rewards as base

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def smoothstep(value: torch.Tensor) -> torch.Tensor:
  """C1-continuous unit ramp, clamped outside [0, 1]."""
  t = value.clamp(0.0, 1.0)
  return t.square() * (3.0 - 2.0 * t)


def command_gate(env, command_name: str, command_threshold: float = 0.05):
  """Original linear-norm plus absolute-yaw command magnitude, smoothly gated."""
  if command_threshold <= 0:
    raise ValueError("command_threshold must be positive")
  command = env.command_manager.get_command(command_name)
  assert command is not None
  speed = command[:, :2].norm(dim=1) + command[:, 2].abs()
  return smoothstep(speed / command_threshold)


def track_linear_velocity_fine(
  env,
  command_name: str,
  std_xy: float = 0.10,
  std_z: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Narrow body-frame XY tracking; retain original vertical tolerance."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = env.scene[asset_cfg.name].data.root_link_lin_vel_b
  error = ((command[:, :2] - actual[:, :2]) / std_xy).square().sum(1)
  return torch.exp(-error - (actual[:, 2] / std_z).square())


def track_angular_velocity_fine(
  env,
  command_name: str,
  std_yaw: float = 0.05,
  std_xy: float = math.sqrt(0.5),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Narrow body-frame yaw tracking; retain original roll/pitch tolerance."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = env.scene[asset_cfg.name].data.root_link_ang_vel_b
  error = ((command[:, 2] - actual[:, 2]) / std_yaw).square()
  return torch.exp(-error - (actual[:, :2] / std_xy).square().sum(1))


def huber(error: torch.Tensor, delta: float) -> torch.Tensor:
  """Unnormalized Huber: 0.5 e² inside delta, delta (|e|-0.5 delta) outside."""
  if delta <= 0:
    raise ValueError("delta must be positive")
  magnitude = error.abs()
  quadratic = magnitude.clamp(max=delta)
  return 0.5 * quadratic.square() + delta * (magnitude - quadratic)


def linear_velocity_huber_cost(
  env,
  command_name: str,
  delta: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive sum of planar body-frame X/Y Huber costs; no added Z penalty."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = env.scene[asset_cfg.name].data.root_link_lin_vel_b
  return huber(command[:, :2] - actual[:, :2], delta).sum(1)


def yaw_velocity_huber_cost(
  env,
  command_name: str,
  delta: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive body-frame yaw Huber cost (no added roll/pitch cost)."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = env.scene[asset_cfg.name].data.root_link_ang_vel_b
  return huber(command[:, 2] - actual[:, 2], delta)


def feet_clearance(
  env,
  target_height,
  height_sensor_name,
  command_name=None,
  command_threshold=0.05,
  asset_cfg=_DEFAULT_ASSET_CFG,
):
  cost = base.feet_clearance(
    env, target_height, height_sensor_name, command_name, -1.0, asset_cfg
  )
  if command_name is None:
    return cost
  return cost * command_gate(env, command_name, command_threshold)


class feet_swing_height(base.feet_swing_height):
  """Original state machine, called once per step, with a smooth command gate."""

  def __call__(
    self,
    env,
    sensor_name,
    height_sensor_name,
    target_height,
    command_name,
    command_threshold=0.05,
  ):
    cost = super().__call__(
      env, sensor_name, height_sensor_name, target_height, command_name, -1.0
    )
    return cost * command_gate(env, command_name, command_threshold)


def feet_slip(
  env,
  sensor_name,
  command_name,
  command_threshold=0.05,
  asset_cfg=_DEFAULT_ASSET_CFG,
):
  cost = base.feet_slip(env, sensor_name, command_name, -1.0, asset_cfg)
  return cost * command_gate(env, command_name, command_threshold)


def soft_landing(
  env,
  sensor_name,
  command_name=None,
  command_threshold=0.05,
):
  cost = base.soft_landing(env, sensor_name, command_name, -1.0)
  if command_name is None:
    return cost
  return cost * command_gate(env, command_name, command_threshold)


class variable_posture(base.variable_posture):
  """Interpolate original per-joint stds at 0/walk/run speed anchors.

  Keep exp(-mean(error²/std²)), joint selection and original initialization.
  Standing tolerance is exact at zero, walking/running at their thresholds.
  Both interpolation intervals have zero endpoint slopes.
  """

  def __call__(
    self,
    env,
    std_standing,
    std_walking,
    std_running,
    asset_cfg,
    command_name,
    walking_threshold=0.5,
    running_threshold=1.5,
  ):
    del std_standing, std_walking, std_running
    if not 0 < walking_threshold < running_threshold:
      raise ValueError("Require 0 < walking_threshold < running_threshold")
    command = env.command_manager.get_command(command_name)
    assert command is not None
    speed = command[:, :2].norm(dim=1) + command[:, 2].abs()
    walk = smoothstep(speed / walking_threshold).unsqueeze(1)
    run = smoothstep(
      (speed - walking_threshold) / (running_threshold - walking_threshold)
    ).unsqueeze(1)
    std = (
      self.std_standing
      + walk * (self.std_walking - self.std_standing)
      + run * (self.std_running - self.std_walking)
    )
    actual = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    desired = self.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-((actual - desired) / std).square().mean(1))


def configure_low_rewards(cfg) -> None:
  """Replace only candidate terms, retaining pose/feet/action-rate weights.

  Match original callables rather than relying on reward dictionary key names.
  Call once when constructing a fresh original environment configuration.
  """
  replacements = {
    base.variable_posture: variable_posture,
    base.feet_clearance: feet_clearance,
    base.feet_swing_height: feet_swing_height,
    base.feet_slip: feet_slip,
    base.soft_landing: soft_landing,
  }
  additions = {}
  for name, term in cfg.rewards.items():
    if term.func in replacements:
      original = term.func
      term.func = replacements[original]
      if original is not base.variable_posture:
        term.params["command_threshold"] = 0.05
    elif term.func in (base.track_linear_velocity, base.track_angular_velocity):
      term.weight = 0.5
      linear = term.func is base.track_linear_velocity
      params = {k: v for k, v in term.params.items() if k != "std"}
      additions[name + "_fine"] = RewardTermCfg(
        func=track_linear_velocity_fine if linear else track_angular_velocity_fine,
        weight=1.5,
        params=params,
      )
  cfg.rewards.update(additions)
