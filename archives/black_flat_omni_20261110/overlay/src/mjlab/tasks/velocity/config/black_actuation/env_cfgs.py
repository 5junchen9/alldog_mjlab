"""Independent motor-friction / action-delay ablations of Black Stock."""

from dataclasses import fields

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.config.black_stock.env_cfgs import (
  black_stock_flat_env_cfg,
  black_stock_rough_env_cfg,
)

from .actuation import BlackDelayedJointPositionActionCfg, BlackFrictionActuatorCfg


def _adapt_actuation(
  cfg: ManagerBasedRlEnvCfg,
  *,
  friction: bool,
  control_delay: bool,
  clear_history_on_reset: bool,
) -> ManagerBasedRlEnvCfg:
  if friction:
    articulation = cfg.scene.entities["robot"].articulation
    assert articulation is not None
    actuator = articulation.actuators[0]
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    assert actuator.effort_limit is not None
    articulation.actuators = (
      BlackFrictionActuatorCfg(
        target_names_expr=actuator.target_names_expr,
        stiffness=actuator.stiffness,
        damping=actuator.damping,
        effort_limit=actuator.effort_limit,
      ),
    )
  if control_delay:
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    cfg.actions["joint_pos"] = BlackDelayedJointPositionActionCfg(
      **{field.name: getattr(action, field.name) for field in fields(action)},
      clear_history_on_reset=clear_history_on_reset,
    )
  return cfg


def black_actuation_flat_env_cfg(
  play: bool = False,
  *,
  friction: bool = True,
  control_delay: bool = True,
  clear_history_on_reset: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _adapt_actuation(
    black_stock_flat_env_cfg(play=play),
    friction=friction,
    control_delay=control_delay,
    clear_history_on_reset=clear_history_on_reset,
  )


def black_actuation_rough_env_cfg(
  play: bool = False,
  *,
  friction: bool = True,
  control_delay: bool = True,
  clear_history_on_reset: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _adapt_actuation(
    black_stock_rough_env_cfg(play=play),
    friction=friction,
    control_delay=control_delay,
    clear_history_on_reset=clear_history_on_reset,
  )
