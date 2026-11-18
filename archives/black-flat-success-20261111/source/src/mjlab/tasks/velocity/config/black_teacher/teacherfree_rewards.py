"""Applied-command-conditioned rewards for one shared teacher-free policy.

All kernels return unweighted, instantaneous values. RewardManager owns weights
and time integration. Gates use only the currently applied command, never the
sampled target or hidden sampling profile. Original common rewards are untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import rewards as base

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def smoothstep(value: torch.Tensor) -> torch.Tensor:
  """Clamped C1 unit ramp (the max/norm command reduction need not be C1)."""
  t = value.clamp(0.0, 1.0)
  return t.square() * (3.0 - 2.0 * t)


def command_gates(command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Return low and high gates from the applied body-frame XY/yaw command."""
  planar = command[:, :2].norm(dim=1)
  yaw = command[:, 2].abs()
  low_scale = torch.maximum(planar / 0.5, yaw / 0.3)
  high_scale = torch.maximum(planar / 1.0, yaw / 0.5)
  return (
    1.0 - smoothstep((low_scale - 0.9) / 0.3),
    smoothstep((high_scale - 0.5) / 0.5),
  )


def command_condition_masks(command: torch.Tensor) -> dict[str, torch.Tensor]:
  """Disjoint logging masks, not latent profile labels or reward switches.

  Low: full low gate; high: full high gate; transition: everything else.
  These masks describe the applied command, not the sampler's intended target.
  """
  low, high = command_gates(command)
  low_mask = low == 1.0
  high_mask = high == 1.0
  return {"low": low_mask, "transition": ~(low_mask | high_mask), "high": high_mask}


def _command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = _command(env, command_name)
  alpha = 0.5 * command_gates(command)[0]
  actual = env.scene[asset_cfg.name].data.root_link_lin_vel_b
  xy_error = (command[:, :2] - actual[:, :2]).square().sum(1)
  z_error = actual[:, 2].square()
  broad = torch.exp(-(xy_error + z_error) / 0.25)
  fine = torch.exp(-xy_error / 0.01 - z_error / 0.25)
  return (1.0 - alpha) * broad + alpha * fine


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = _command(env, command_name)
  alpha = 0.5 * command_gates(command)[0]
  actual = env.scene[asset_cfg.name].data.root_link_ang_vel_b
  yaw_error = (command[:, 2] - actual[:, 2]).square()
  xy_error = actual[:, :2].square().sum(1)
  broad = torch.exp(-(yaw_error + xy_error) / 0.5)
  fine = torch.exp(-yaw_error / 0.0064 - xy_error / 0.5)
  return (1.0 - alpha) * broad + alpha * fine


def huber(error: torch.Tensor, delta: float) -> torch.Tensor:
  """Unnormalized Huber: .5e² if |e| <= delta, else delta(|e|-.5delta)."""
  if delta <= 0:
    raise ValueError("delta must be positive")
  magnitude = error.abs()
  quadratic = magnitude.clamp(max=delta)
  return 0.5 * quadratic.square() + delta * (magnitude - quadratic)


def linear_velocity_huber_cost(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = _command(env, command_name)
  actual = env.scene[asset_cfg.name].data.root_link_lin_vel_b
  return command_gates(command)[1] * huber(command[:, :2] - actual[:, :2], 0.5).sum(1)


def yaw_velocity_huber_cost(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = _command(env, command_name)
  actual = env.scene[asset_cfg.name].data.root_link_ang_vel_b
  return command_gates(command)[1] * huber(command[:, 2] - actual[:, 2], 0.25)


def configure_teacherfree_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace exactly the two original tracking terms and add two Huber costs.

  Require a fresh original config so accidental candidate/double configuration
  fails before mutation. Preserve every other reward object, parameter and weight.
  """
  linear = [k for k, t in cfg.rewards.items() if t.func is base.track_linear_velocity]
  angular = [k for k, t in cfg.rewards.items() if t.func is base.track_angular_velocity]
  if len(linear) != 1 or len(angular) != 1:
    raise ValueError("Expected exactly one original linear and angular tracking term")
  additions = ("teacherfree_linear_huber", "teacherfree_yaw_huber")
  if any(name in cfg.rewards for name in additions):
    raise ValueError("Teacher-free Huber reward names already exist")
  for name, func, cost, cost_name, weight in (
    (linear[0], track_linear_velocity, linear_velocity_huber_cost, additions[0], -0.20),
    (angular[0], track_angular_velocity, yaw_velocity_huber_cost, additions[1], -0.10),
  ):
    params = {k: v for k, v in cfg.rewards[name].params.items() if k != "std"}
    cfg.rewards[name] = RewardTermCfg(func=func, weight=2.0, params=params)
    cfg.rewards[cost_name] = RewardTermCfg(
      func=cost, weight=weight, params=dict(params)
    )
