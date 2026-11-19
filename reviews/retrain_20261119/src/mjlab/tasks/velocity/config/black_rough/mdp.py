"""Domain-aware R0 rewards, failure priority and checkpointable local curriculum."""

from collections import deque
from copy import deepcopy
from typing import Any

import torch

from mjlab.envs import mdp as common
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity
from mjlab.tasks.velocity.mdp.rewards import upright
from mjlab.utils.lab_api.math import quat_apply_inverse


def flat_mask(env):
  return env.command_manager.get_term("twist").flat_mask


def contact_magnitude(sensor):
  # Trunk persistence is based on the current force, not a trailing history max.
  return sensor.data.force.norm(dim=-1).amax(dim=-1)


def patch_exit_decision(
  displacement, world_command, progress, demand, failed, flat, steps, coverage
):
  edge = (displacement.abs().amax(-1) > 3.6) & (~flat) & (steps > 2)
  intended = (
    ((displacement * world_command).sum(-1) > 0)
    & (progress >= 0.8 * demand)
    & (demand >= 1.0)
    & (coverage >= 0.95)
  )
  success = edge & intended & ~failed
  wrong_exit = edge & ~intended
  return success, wrong_exit


_DEFAULT_ASSET = SceneEntityCfg("robot")


class MixedUpright(upright):
  def __call__(self, env, std, asset_cfg=_DEFAULT_ASSET, terrain_sensor_names=None):
    rough = super().__call__(env, std, asset_cfg, terrain_sensor_names)
    robot = env.scene[asset_cfg.name]
    quat = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    gravity = quat_apply_inverse(quat, robot.data.gravity_vec_w)
    flat = torch.exp(-gravity[:, :2].square().sum(-1) / std**2)
    return torch.where(flat_mask(env), flat, rough)


def rough_collision(env, sensor_name):
  return velocity.self_collision_cost(env, sensor_name)


def root_clearance(env):
  robot = env.scene["robot"]
  scan = env.scene["terrain_scan"].data
  hits = scan.hit_pos_w
  near = (hits[:, :, :2] - robot.data.root_link_pos_w[:, None, :2]).norm(dim=-1) < 0.25
  valid = (scan.distances >= 0) & near
  ground = torch.where(valid, hits[:, :, 2], -torch.inf).amax(-1)
  # No miss fallback to world zero: a missing support estimate remains invalid.
  return robot.data.root_link_pos_w[:, 2] - ground, valid.any(-1)


class FailureState:
  def __init__(self, env):
    self.env = env
    n, device = env.num_envs, env.device
    # Columns: 0=thigh, 1=trunk, 2=tilt, 3=low_clearance (keeps -1 == clearance).
    self.age = torch.zeros(n, 4, device=device)
    self.flat_low_age = torch.zeros(n, device=device)
    self.failed = torch.zeros(n, dtype=torch.bool, device=device)
    self.reason = torch.zeros(n, 4, dtype=torch.bool, device=device)
    self.last_xy = torch.zeros(n, 2, device=device)
    self.progress = torch.zeros(n, device=device)
    self.demand = torch.zeros(n, device=device)
    self.valid_steps = torch.zeros(n, device=device)
    self.motion_steps = torch.zeros(n, device=device)
    self.last_step = -1
    self.enforce_patch_bounds = True
    self.traversal = torch.zeros(n, dtype=torch.bool, device=device)
    self.maximum_level = 1
    self.type_ceilings = [0, 1, 1, 1, 1, 1, 1]
    self.audit_streak = 0
    self.histories = [deque(maxlen=50) for _ in range(7)]
    self.failure_streak = torch.zeros(n, dtype=torch.long, device=device)
    self.sensor_misses = torch.zeros(4, dtype=torch.long, device=device)

  def update(self):
    env = self.env
    if self.last_step == env.common_step_counter:
      return
    self.last_step = env.common_step_counter
    robot = env.scene["robot"]
    xy = robot.data.root_link_pos_w[:, :2]
    delta = xy - self.last_xy
    delta[env.episode_length_buf <= 1] = 0
    self.last_xy.copy_(xy)
    command = env.command_manager.get_term("twist").command
    speed = command[:, :2].norm(dim=-1)
    heading = robot.data.heading_w
    c, s = heading.cos(), heading.sin()
    world = torch.stack(
      (c * command[:, 0] - s * command[:, 1], s * command[:, 0] + c * command[:, 1]), -1
    )
    direction = world / speed.clamp(min=1e-8)[:, None]
    self.progress += (delta * direction).sum(-1)
    self.demand += speed * env.step_dt
    self.motion_steps += speed > 0.04
    foot = env.scene["foot_height_scan"].data
    foot_valid = (foot.distances.reshape(env.num_envs, 4, -1) >= 0).any(-1)
    self.sensor_misses += (~foot_valid).sum(0)
    valid = foot_valid.all(-1) & torch.isfinite(foot.heights).all(-1)
    self.valid_steps += valid & (speed > 0.04)
    angle = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
    clearance, support = root_clearance(env)
    thigh = velocity.illegal_contact(env, "thigh_ground_touch")
    trunk_touch = contact_magnitude(env.scene["trunk_ground_touch"]) > 10
    persistent = torch.stack(
      (
        thigh,
        trunk_touch,
        angle > torch.pi * 70 / 180,
        support & (clearance < 0.15),
      ),
      -1,
    )
    self.age = torch.where(persistent, self.age + env.step_dt, 0)
    self.flat_low_age = torch.where(
      support & (clearance < 0.12), self.flat_low_age + env.step_dt, 0
    )
    thresholds = torch.tensor(
      [0.15 - 1e-6, 0.10 - 1e-6, 0.10 - 1e-6, 0.10 - 1e-6], device=self.age.device
    )
    rough_reason = self.age >= thresholds
    flat_reason = torch.stack(
      (
        torch.zeros_like(thigh),
        self.age[:, 1] >= 0.10 - 1e-6,
        angle > torch.pi * 70 / 180,
        self.flat_low_age >= 0.15 - 1e-6,
      ),
      -1,
    )
    self.reason = torch.where(flat_mask(env)[:, None], flat_reason, rough_reason)
    self.failed = self.reason.any(-1)
    self.failed |= ~torch.isfinite(robot.data.root_link_pos_w).all(-1)
    if self.enforce_patch_bounds:
      displacement = xy - env.scene.env_origins[:, :2]
      self.traversal, wrong_exit = patch_exit_decision(
        displacement,
        world,
        self.progress,
        self.demand,
        self.failed,
        flat_mask(env),
        env.episode_length_buf,
        self.valid_steps / self.motion_steps.clamp(min=1),
      )
      self.failed |= wrong_exit

  def reset(self, ids):
    for name in [
      "age",
      "flat_low_age",
      "progress",
      "demand",
      "valid_steps",
      "motion_steps",
    ]:
      getattr(self, name)[ids] = 0
    self.failed[ids] = False
    self.traversal[ids] = False
    self.reason[ids] = False

  def state_dict(self):
    names = [
      "age",
      "flat_low_age",
      "failed",
      "reason",
      "last_xy",
      "progress",
      "demand",
      "valid_steps",
      "motion_steps",
      "failure_streak",
      "sensor_misses",
      "traversal",
    ]
    return {
      "tensors": {k: getattr(self, k).cpu().clone() for k in names},
      "maximum_level": self.maximum_level,
      "type_ceilings": self.type_ceilings,
      "audit_streak": self.audit_streak,
      "histories": [list(x) for x in self.histories],
    }

  def load_state_dict(self, state):
    for k, value in state["tensors"].items():
      getattr(self, k).copy_(value.to(self.env.device))
    self.maximum_level = state["maximum_level"]
    self.type_ceilings = state["type_ceilings"]
    self.audit_streak = state["audit_streak"]
    self.histories = [deque(x, maxlen=50) for x in state["histories"]]
    self.last_step = -1


def state(env) -> FailureState:
  if not hasattr(env, "black_rough_state"):
    env.black_rough_state = FailureState(env)
  return env.black_rough_state


def failures(env):
  result = state(env)
  result.update()
  return result.failed


def patch_success(env):
  result = state(env)
  result.update()
  return result.traversal & ~result.failed


def timeout(env):
  return common.time_out(env) & ~failures(env)


def map_bounds(env):
  return velocity.out_of_terrain_bounds(env) & ~flat_mask(env) & ~failures(env)


def curriculum(env, env_ids):
  st = state(env)
  terrain = env.scene.terrain
  if env.common_step_counter == 0:
    return {"maximum_level": torch.tensor(1.0, device=env.device)}
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  for index in ids.cpu().tolist():
    if flat_mask(env)[index] or env.episode_length_buf[index] == 0:
      continue
    kind = int(terrain.terrain_types[index])
    moving = bool(st.demand[index] >= 1.0)
    successful = bool(
      not st.failed[index]
      and st.progress[index] >= 0.8 * st.demand[index]
      and st.valid_steps[index] >= 0.95 * st.motion_steps[index]
    )
    st.failure_streak[index] = st.failure_streak[index] + 1 if st.failed[index] else 0
    if moving:
      st.histories[kind].append(successful)
    history = st.histories[kind]
    promote = moving and len(history) == 50 and sum(history) >= 40 and successful
    demote = st.failure_streak[index] >= 3 or (len(history) == 50 and sum(history) < 30)
    step = int(promote) - int(demote)
    terrain.terrain_levels[index] = (terrain.terrain_levels[index] + step).clamp(
      1, st.type_ceilings[kind]
    )
  terrain.env_origins[ids] = terrain.terrain_origins[
    terrain.terrain_levels[ids], terrain.terrain_types[ids]
  ]
  return {"maximum_level": torch.tensor(float(st.maximum_level), device=env.device)}


def reset_base(env, env_ids, pose_range, velocity_range):
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  flat = ids[flat_mask(env)[ids]]
  rough = ids[~flat_mask(env)[ids]]
  if len(flat):
    common.reset_root_state_uniform(env, flat, pose_range, velocity_range)
  if len(rough):
    ranges = deepcopy(pose_range)
    ranges["x"], ranges["y"] = (-0.2, 0.2), (-0.2, 0.2)
    common.reset_root_state_uniform(env, rough, ranges, velocity_range)
  state(env).reset(ids)


def update_audit_curriculum(st: Any, rough: dict, flat_passed: bool):
  """Independent audit controls global ceiling; highest row never randomizes."""
  groups = rough["groups"]
  passed = flat_passed and all(
    g["success_fraction"] >= 0.95
    and g["fall_fraction"] <= 0.01
    and g["valid_fraction"] >= 0.95
    and g["minimum_direction_success"] >= 0.9
    for g in groups
  )
  st.audit_streak = st.audit_streak + 1 if passed else 0
  if st.audit_streak >= 2:
    st.maximum_level = min(3, st.maximum_level + 1)
    st.audit_streak = 0
  return passed
