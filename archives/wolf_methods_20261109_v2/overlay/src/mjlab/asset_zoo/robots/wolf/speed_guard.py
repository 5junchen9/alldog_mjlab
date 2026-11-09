"""Experimental software speed guard, not a measured Wolf motor model.

Clamps commanded velocity and suppresses torque that would drive farther into an
overspeed condition. It never edits simulated position or velocity. External
forces, inertia and discrete integration can still produce overspeed. No torque-
speed curve, active braking gain, electrical constants or thermal model is added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd, TransmissionType
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class WolfSpeedGuardActuatorCfg(IdealPdActuatorCfg):
  """Opt-in software guard using a declared URDF joint speed in rad/s.

  This threshold is not a guarantee on physical state or motor-controller
  behavior. Preserve the original IdealPd profile for existing policies.
  """

  velocity_limit: float

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.transmission_type != TransmissionType.JOINT:
      raise ValueError("Wolf speed guard only supports joint transmission")
    if not math.isfinite(self.velocity_limit) or self.velocity_limit <= 0:
      raise ValueError("velocity_limit must be a positive finite speed in rad/s")

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> WolfSpeedGuardActuator:
    return WolfSpeedGuardActuator(self, entity, target_ids, target_names)


class WolfSpeedGuardActuator(IdealPdActuator[WolfSpeedGuardActuatorCfg]):
  """Guard delayed targets using current, undelayed joint-speed feedback."""

  param_names = (*IdealPdActuator.param_names, "software_velocity_limit")

  def __init__(
    self,
    cfg: WolfSpeedGuardActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.software_velocity_limit: torch.Tensor | None = None

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    self.software_velocity_limit = torch.full(
      (data.nworld, len(self._target_names)),
      self.cfg.velocity_limit,
      dtype=torch.float,
      device=device,
    )

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    limit = params["software_velocity_limit"]
    guarded_command = ActuatorCmd(
      position_target=cmd.position_target,
      velocity_target=torch.clamp(cmd.velocity_target, -limit, limit),
      effort_target=cmd.effort_target,
      pos=cmd.pos,
      vel=cmd.vel,
    )
    torque = IdealPdActuator.control_law(params, guarded_command)
    drives_overspeed = ((cmd.vel >= limit) & (torque > 0)) | (
      (cmd.vel <= -limit) & (torque < 0)
    )
    return torch.where(drives_overspeed, torch.zeros_like(torque), torque)
