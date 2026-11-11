"""Opt-in Black control laws from super-dog@1b9f15de, not its RL settings.

Black overrides the base LeggedRobot implementation: friction is smooth tanh,
lag is sampled at reset, and action history is NOT cleared on episode reset.
The independent stock tasks continue to use their original built-in servo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import (
  IdealPdActuator,
  IdealPdActuatorCfg,
  pd_torque,
)
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BlackFrictionActuatorCfg(IdealPdActuatorCfg):
  """Explicit PD minus source motor friction, then effort saturation."""

  friction_enabled: bool = True

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> BlackFrictionActuator:
    return BlackFrictionActuator(self, entity, target_ids, target_names)


class BlackFrictionActuator(IdealPdActuator[BlackFrictionActuatorCfg]):
  """Uses current state at every physics substep; no delayed feedback."""

  param_names = (*IdealPdActuator.param_names, "friction_scale")

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    assert self.stiffness is not None
    self.friction_scale = torch.full_like(
      self.stiffness, float(self.cfg.friction_enabled)
    )

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    torque = pd_torque(params["stiffness"], params["damping"], cmd)
    friction = 0.35 * torch.tanh(3.0 * cmd.vel) + 0.1 * cmd.vel
    torque = torque - params["friction_scale"] * friction
    limit = params["force_limit"]
    return torch.clamp(torque, -limit, limit)


class ControlStepDelay:
  """One history shift per policy/control step, never per physics substep.

  Source initializes four slots and zero lag, samples U{0,1,2} only on reset,
  and retains the previous episode's history. Optional clearing is an explicit
  deviation for episode-isolated experiments, not the source-compatible default.
  """

  def __init__(self, num_envs: int, action_dim: int, device: str):
    self.history = torch.zeros(num_envs, 4, action_dim, device=device)
    self.lags = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._env_ids = torch.arange(num_envs, device=device)

  def reset(
    self,
    env_ids: torch.Tensor | slice | None = None,
    *,
    enabled: bool = True,
    clear_history: bool = False,
  ) -> None:
    ids = slice(None) if env_ids is None else env_ids
    if enabled:
      self.lags[ids] = torch.randint(
        0, 3, self.lags[ids].shape, device=self.lags.device
      )
    if clear_history:
      self.history[ids] = 0.0

  def push(self, actions: torch.Tensor, *, enabled: bool = True) -> torch.Tensor:
    self.history[:, 1:] = self.history[:, :-1].clone()
    self.history[:, 0] = actions
    return self.history[self._env_ids, self.lags] if enabled else actions


@dataclass(kw_only=True)
class BlackDelayedJointPositionActionCfg(JointPositionActionCfg):
  """Delay actions by a reset-sampled number of complete control steps."""

  delay_enabled: bool = True
  clear_history_on_reset: bool = False
  """False matches pinned Black; True prevents cross-episode action carryover."""

  def build(self, env: ManagerBasedRlEnv) -> BlackDelayedJointPositionAction:
    return BlackDelayedJointPositionAction(self, env)


class BlackDelayedJointPositionAction(JointPositionAction):
  # This subclass is constructed only with its matching configuration.
  cfg: BlackDelayedJointPositionActionCfg  # pyright: ignore

  def __init__(
    self, cfg: BlackDelayedJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    self.control_delay = ControlStepDelay(self.num_envs, self.action_dim, self.device)

  def process_actions(self, actions: torch.Tensor) -> None:
    delayed = self.control_delay.push(actions, enabled=self.cfg.delay_enabled)
    super().process_actions(delayed)
    # Observation/action-rate histories remain current-policy actions, as in
    # stock mjlab and source Black. Only the command sent to PD is delayed.
    self._raw_actions.copy_(actions)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self.control_delay.reset(
      env_ids,
      enabled=self.cfg.delay_enabled,
      clear_history=self.cfg.clear_history_on_reset,
    )
