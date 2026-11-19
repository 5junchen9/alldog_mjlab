"""Versioned 72-dimensional no-forward-vision T2 environment."""

from copy import deepcopy
from dataclasses import fields

from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.sensor import RayCastSensorCfg
from mjlab.tasks.velocity.config.black_actuation.env_cfgs import (
  black_actuation_rough_env_cfg,
)
from mjlab.tasks.velocity.config.black_teacher.teacherfree_env_cfg import (
  black_teacherfree_env_cfg,
)
from mjlab.tasks.velocity.config.black_teacher.teacherfree_rewards import (
  configure_teacherfree_rewards,
)

from . import mdp
from .command import RoughVelocityCommandCfg
from .terrain_cfgs import terrain_config

TASK = "Mjlab-Velocity-Rough-Black-Teacher-72-v2"
OBSERVATION_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command",
  "foot_height",
  "foot_air_time",
  "foot_contact",
  "foot_contact_forces",
)


def validate_contract(cfg):
  reference = black_teacherfree_env_cfg()
  for group in ("actor", "critic"):
    if tuple(cfg.observations[group].terms) != OBSERVATION_TERMS:
      raise ValueError(f"{group}: 72-dimensional observation order/semantics changed")
    for name in OBSERVATION_TERMS:
      actual = cfg.observations[group].terms[name]
      expected = reference.observations[group].terms[name]
      if (
        actual.func is not expected.func
        or actual.params != expected.params
        or actual.scale != expected.scale
      ):
        raise ValueError(f"{group}/{name}: same-width semantic change")
    if cfg.observations[group].enable_corruption:
      raise ValueError("True-state teacher must not corrupt observations")
  if cfg.actions["joint_pos"].scale != 0.25:
    raise ValueError("Parent action scale changed")


def rough_env_cfg(num_envs=4096, seed=122001, map_seed=112001, play=False):
  cfg = black_actuation_rough_env_cfg(play=False)
  cfg.scene.num_envs, cfg.seed = num_envs, seed
  cfg.sim.ccd_scratch_slots = 1024
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = terrain_config(map_seed)
  cfg.scene.terrain.max_init_terrain_level = 1
  del cfg.observations["critic"].terms["height_scan"]
  cfg.observations["actor"] = deepcopy(cfg.observations["critic"])
  cfg.observations["actor"].enable_corruption = False
  cfg.observations["critic"].enable_corruption = False
  old = cfg.commands["twist"]
  cfg.commands["twist"] = RoughVelocityCommandCfg(
    **{f.name: deepcopy(getattr(old, f.name)) for f in fields(old)}, profile_seed=seed
  )
  cfg.commands["twist"].debug_vis = False
  cfg.commands["twist"].heading_command = False
  cfg.commands["twist"].ranges.heading = None
  cfg.commands["twist"].rel_heading_envs = 0
  cfg.commands["twist"].rel_forward_envs = 0
  cfg.commands["twist"].rel_standing_envs = 0
  configure_teacherfree_rewards(cfg)
  cfg.rewards["upright"].func = mdp.MixedUpright
  for name in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards[name].func = mdp.rough_collision
  cfg.events["reset_base"].func = mdp.reset_base
  cfg.terminations = {
    "failure": TerminationTermCfg(func=mdp.failures),
    "time_out": TerminationTermCfg(func=mdp.timeout, time_out=True),
    "out_of_map": TerminationTermCfg(func=mdp.map_bounds, time_out=True),
    "patch_traversal": TerminationTermCfg(func=mdp.patch_success, time_out=True),
  }
  cfg.curriculum = {"rough_levels": CurriculumTermCfg(func=mdp.curriculum)}
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, RayCastSensorCfg):
      sensor.debug_vis = False
  validate_contract(cfg)
  return cfg
