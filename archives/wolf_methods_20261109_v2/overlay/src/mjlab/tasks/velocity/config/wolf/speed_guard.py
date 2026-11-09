"""Explicit opt-in Wolf speed-guard configuration; no task defaults are changed."""

from copy import deepcopy
from dataclasses import fields
from xml.etree import ElementTree as ET

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.asset_zoo.robots.wolf.speed_guard import WolfSpeedGuardActuatorCfg
from mjlab.asset_zoo.robots.wolf.wolf_constants import (
  WOLF_POLICY_JOINT_NAMES,
  WOLF_XML,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg

_WOLF_TASKS = ("wolf-flat", "wolf-rough", "wolf-flat-him", "wolf-rough-him")


def with_wolf_urdf_speed_guard(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Return an independent experimental config with declared-speed guarding.

  Reads thresholds from the pinned local design URDF. Original gains, torque
  caps, delays, observation/action contracts and physical state are unchanged.
  The original config is neither mutated nor registered under a new default.
  """
  result = deepcopy(cfg)
  robot = result.scene.entities["robot"]
  if robot.articulation is None:
    raise ValueError("Wolf articulation is required")
  original = robot.articulation.actuators
  names = tuple(name for act in original for name in act.target_names_expr)
  if len(original) != 16 or set(names) != set(WOLF_POLICY_JOINT_NAMES):
    raise ValueError("Expected exactly the 16 explicit Wolf joint actuators")
  urdf = ET.parse(WOLF_XML.parent.parent / "urdf/wolf_description.urdf").getroot()
  limits = {}
  for joint in urdf.findall("joint"):
    name = joint.get("name")
    if name in WOLF_POLICY_JOINT_NAMES:
      limit = joint.find("limit")
      if limit is None or "velocity" not in limit.attrib:
        raise ValueError(f"Missing source velocity limit for {name}")
      limits[name] = float(limit.attrib["velocity"])
  guarded = []
  for act in original:
    if type(act) is not IdealPdActuatorCfg or len(act.target_names_expr) != 1:
      raise ValueError("Only an original per-joint IdealPd Wolf profile is supported")
    name = act.target_names_expr[0]
    if name not in limits:
      raise ValueError(f"Missing source joint {name}")
    values = {field.name: deepcopy(getattr(act, field.name)) for field in fields(act)}
    guarded.append(WolfSpeedGuardActuatorCfg(**values, velocity_limit=limits[name]))
  robot.articulation.actuators = tuple(guarded)
  return result


def wolf_speed_guard_env_cfg(
  task_id: str, *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build an opt-in experiment beside one of the four original Wolf tasks."""
  if task_id not in _WOLF_TASKS:
    raise ValueError(f"Expected one of {_WOLF_TASKS}, got {task_id!r}")
  return with_wolf_urdf_speed_guard(load_env_cfg(task_id, play=play))
