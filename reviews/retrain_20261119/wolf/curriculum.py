"""平地起步、通用地形和六专项；专项保留25%平地回放。"""

import math
from dataclasses import dataclass

import torch

from mjlab.asset_zoo.robots.wolf.wolf_constants import (
  WOLF_LEG_JOINT_NAMES,
  WOLF_LEG_ORDER,
  WOLF_WHEEL_BODY_NAMES,
  WOLF_WHEEL_JOINT_NAMES,
)
from mjlab.envs import mdp
from mjlab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity.config.wolf.env_cfg import (
  WOLF_WHEEL_SLIP_CONTACT_SENSOR,
  make_env_cfg,
)
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.terrains import BoxFlatTerrainCfg, TerrainGeneratorCfg
from wolf import rewards

SPECIALTIES = (
  "stairs",
  "platform",
  "high_wall",
  "slope",
  "broken_bridge",
  "plum_stakes",
)
COURSES = ("flat_foundation", "flat_axes", "general", *SPECIALTIES)
SETTLE_CONTROLS = 50


def flat_command(tick, target, general=False):
  """先静止1秒，再低速起动、停、反向、停；命令不经过旧0.2m/s死区。"""
  phase = tick % (1350 if general else 550)
  first_end, second_start, second_end = (550, 700, 1200) if general else (200, 300, 450)
  result = torch.zeros_like(target)
  result = torch.where(((phase >= 50) & (phase < first_end))[:, None], target, result)
  result = torch.where(
    ((phase >= second_start) & (phase < second_end))[:, None], -target, result
  )
  return result


class CourseCommand(UniformVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.course = cfg.course
    self.target = torch.zeros(self.num_envs, 3, device=self.device)
    self.flat_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self.direction = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
    self.settled = torch.zeros_like(self.direction)
    self.passed = torch.zeros_like(self.flat_mask)
    self.failed = torch.zeros_like(self.flat_mask)
    self.entered = torch.zeros_like(self.flat_mask)
    self.contact_steps = torch.zeros_like(self.direction)
    self.last_tick = -1
    self.lengths = self.widths = None
    self.cases = []
    self.unlocked_level = 0
    self.promotion_successes = 0
    self.promotion_results = []
    self.sparse_catalog = None
    self.treads = []
    self.ordered = torch.zeros_like(self.direction)
    self.touched = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
    if self.course in SPECIALTIES:
      from support.black.obstacle_geometry import GROUPS, historical_case

      self.cases = [historical_case(name)[2] for name in GROUPS[self.course]]
      self.lengths = torch.tensor(
        [c["xmax"] - c["xmin"] + 1.5 for c in self.cases], device=self.device
      )
      self.widths = torch.tensor(
        [c["half_width"] for c in self.cases], device=self.device
      )
      self.flat_mask = env.scene.terrain.terrain_types == 0
      from wolf.eval_score import geometry_contract

      self.treads = [
        geometry_contract({"course": self.course, "case": name, "direction": 1})[
          "treads"
        ]
        for name in GROUPS[self.course]
      ]
      self.tread_counts = torch.tensor(
        [len(rows) for rows in self.treads], device=self.device
      )
      self.tread_bounds = torch.zeros(
        len(self.treads), max(1, max(map(len, self.treads))), 5, device=self.device
      )
      for level, rows in enumerate(self.treads):
        if rows:
          self.tread_bounds[level, : len(rows)] = torch.tensor(rows, device=self.device)
      if self.course in ("broken_bridge", "plum_stakes"):
        from support.black.sparse_support import SparseSupportCatalog

        from mjlab.asset_zoo.robots.wolf.wolf_constants import (
          WOLF_WHEEL_COLLISION_GEOM_NAMES,
        )

        self.sparse_catalog = SparseSupportCatalog(
          env.sim.mj_model, env.device, WOLF_WHEEL_COLLISION_GEOM_NAMES
        )

  def _resample_command(self, env_ids):
    self.target[env_ids] = 0
    course = self.cfg.course
    axis = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
    if course != "flat_foundation":
      axis = torch.randint(3, (len(env_ids),), device=self.device)
    signs = torch.randint(2, (len(env_ids),), device=self.device) * 2 - 1
    magnitudes = torch.where(axis == 2, 0.3, 0.3 if course == "general" else 0.2)
    self.target[env_ids, axis] = signs * magnitudes
    self.target[env_ids[torch.rand(len(env_ids), device=self.device) < 0.25]] = 0

  def reset(self, env_ids):
    result = super().reset(env_ids)
    self.settled[env_ids] = 0
    self.passed[env_ids] = False
    self.failed[env_ids] = False
    self.entered[env_ids] = False
    self.contact_steps[env_ids] = 0
    self.ordered[env_ids] = 0
    self.touched[env_ids] = False
    self.last_tick = -1
    return result

  def _update_command(self, env_ids=None):
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    self.vel_command_b[ids] = flat_command(
      self._env.episode_length_buf[ids], self.target[ids], self.course == "general"
    )
    if self.course not in SPECIALTIES:
      return
    ids = ids[~self.flat_mask[ids]]
    level = self._env.scene.terrain.terrain_levels[ids]
    local = self.robot.data.root_link_pos_w[ids] - self._env.scene.env_origins[ids]
    heading = torch.where(self.direction[ids] > 0, 0.0, math.pi)
    error = torch.atan2(
      torch.sin(heading - self.robot.data.heading_w[ids]),
      torch.cos(heading - self.robot.data.heading_w[ids]),
    )
    progress = torch.where(
      self.direction[ids] > 0, local[:, 0], self.lengths[level] - local[:, 0]
    )
    speed = torch.where(progress < self.lengths[level] - 0.15, 0.2, 0.0)
    speed = torch.where(self._env.episode_length_buf[ids] < 50, 0.0, speed)
    self.vel_command_b[ids, 0] = speed * error.cos()
    self.vel_command_b[ids, 1] = speed * error.sin()
    self.vel_command_b[ids, 2] = torch.where(speed > 0, error.clamp(-0.3, 0.3), 0.0)

  def evidence(self):
    """每控制步一次，在reset前读取材料接触点；底板不能代替桥面/桩顶。"""
    if (
      self.course not in SPECIALTIES or self.last_tick == self._env.common_step_counter
    ):
      return
    self.last_tick = self._env.common_step_counter
    env = self._env
    ids = (~self.flat_mask).nonzero().flatten()
    levels = env.scene.terrain.terrain_levels[ids]
    sensor = env.scene[WOLF_WHEEL_SLIP_CONTACT_SENSOR].data
    points = sensor.pos[ids] - env.scene.env_origins[ids, None, :]
    contacts = (sensor.found[ids] > 0) & (sensor.force[ids, :, 0].abs() > 1.0)
    root = self.robot.data.root_link_pos_w[ids] - env.scene.env_origins[ids]
    forward = self.direction[ids] > 0
    progress = torch.where(
      forward[:, None], points[:, :, 0], self.lengths[levels, None] - points[:, :, 0]
    )
    root_progress = torch.where(forward, root[:, 0], self.lengths[levels] - root[:, 0])
    inside = (root_progress > 0.75) & (root_progress < self.lengths[levels] - 0.75)
    self.entered[ids] |= inside
    self.contact_steps[ids] += (inside & contacts.any(-1)).long()
    self.failed[ids] |= root[:, 1].abs() > self.widths[levels] - 0.05
    # 对所有真实接触逐个对照原碰撞面的箱顶；悬空不是底板接触。
    if self.sparse_catalog is not None:
      sparse = self.sparse_catalog.read(env.sim.data, env.num_envs)
      if sparse["invalid"].any():
        raise FloatingPointError("Wolf桥桩接触非有限")
      self.failed |= sparse["lower_floor"].any(-1)
    if self.course == "stairs":
      unfinished = self.ordered[ids] < self.tread_counts[levels]
      index = torch.where(
        self.direction[ids] > 0,
        self.ordered[ids],
        self.tread_counts[levels] - 1 - self.ordered[ids],
      )
      bounds = self.tread_bounds[levels, index.clamp(0, self.tread_bounds.shape[1] - 1)]
      x0, x1, y0, y1, z = [part[:, None] for part in bounds.unbind(-1)]
      touched = (
        contacts
        & (points[..., 0] >= x0 - 0.02)
        & (points[..., 0] <= x1 + 0.02)
        & (points[..., 1] >= y0 - 0.02)
        & (points[..., 1] <= y1 + 0.02)
        & ((points[..., 2] - z).abs() < 0.03)
      )
      self.touched[ids] |= touched & unfinished[:, None]
      advance = (
        unfinished & self.touched[ids, :2].any(-1) & self.touched[ids, 2:].any(-1)
      )
      self.ordered[ids] += advance.long()
      self.touched[ids[advance]] = False
    stable = (
      (progress > self.lengths[levels, None] - 0.75).all(-1)
      & (progress < self.lengths[levels, None] + 0.65).all(-1)
      & (points[..., 1].abs() < self.widths[levels, None] - 0.02).all(-1)
      & contacts.all(-1)
      & (self.robot.data.root_link_lin_vel_b[ids].norm(dim=-1) < 0.10)
      & (self.robot.data.root_link_ang_vel_b[ids].norm(dim=-1) < 0.10)
      & (torch.acos((-self.robot.data.projected_gravity_b[ids, 2]).clamp(-1, 1)) < 0.20)
    )
    self.settled[ids] = torch.where(stable, self.settled[ids] + 1, 0)
    treads_complete = self.ordered[ids] == self.tread_counts[levels]
    self.passed[ids] |= (
      (self.settled[ids] >= SETTLE_CONTROLS)
      & self.entered[ids]
      & (self.contact_steps[ids] > 0)
      & ~self.failed[ids]
      & treads_complete
    )


@dataclass(kw_only=True)
class CourseCommandCfg(UniformVelocityCommandCfg):
  course: str
  max_level: int = 0
  allow_promotion: bool = True

  def build(self, env):
    return CourseCommand(self, env)


def reset_route(env, env_ids, pose_range, velocity_range):
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  command = env.command_manager.get_term("twist")
  if command.course in SPECIALTIES and command.cfg.allow_promotion:
    # 只用当前最高关最近16个真实attempt，至少14次完整通过才解锁。
    current = env.scene.terrain.terrain_levels[ids] == command.unlocked_level
    attempted = current & ~command.flat_mask[ids] & (env.episode_length_buf[ids] > 0)
    command.promotion_results.extend(command.passed[ids[attempted]].cpu().tolist())
    command.promotion_results = command.promotion_results[-16:]
    command.promotion_successes = sum(command.promotion_results)
    if (
      len(command.promotion_results) == 16
      and command.promotion_successes >= 14
      and command.unlocked_level < command.cfg.max_level
    ):
      command.unlocked_level += 1
      command.promotion_successes = 0
      command.promotion_results = []
    specialty_ids = ids[~command.flat_mask[ids]]
    levels = torch.full_like(specialty_ids, command.unlocked_level)
    replay = torch.rand(len(levels), device=env.device) < 0.25
    levels[replay] = torch.randint(
      command.unlocked_level + 1, (int(replay.sum()),), device=env.device
    )
    env.scene.terrain.terrain_levels[specialty_ids] = levels
    env.scene.terrain.env_origins[specialty_ids] = env.scene.terrain.terrain_origins[
      levels, env.scene.terrain.terrain_types[specialty_ids]
    ]
  mdp.reset_root_state_uniform(env, ids, pose_range, velocity_range)
  if command.course not in SPECIALTIES:
    return
  ids = ids[~command.flat_mask[ids]]
  command.direction[ids] = torch.randint(2, (len(ids),), device=env.device) * 2 - 1
  for level in env.scene.terrain.terrain_levels[ids].unique().tolist():
    for direction in (-1, 1):
      selected = ids[
        (env.scene.terrain.terrain_levels[ids] == level)
        & (command.direction[ids] == direction)
      ]
      offset = float(command.lengths[level]) if direction < 0 else 0.0
      yaw = math.pi if direction < 0 else 0.0
      mdp.reset_root_state_uniform(
        env, selected, {"x": (offset, offset), "yaw": (yaw, yaw)}, {}
      )


def route_finished(env):
  command = env.command_manager.get_term("twist")
  command.evidence()
  return command.passed


def route_failed(env):
  command = env.command_manager.get_term("twist")
  command.evidence()
  return command.failed


def configure(
  course="flat_foundation", count=32, seed=0, max_level=None, allow_promotion=True
):
  if course not in COURSES:
    raise ValueError("未知Wolf课程")
  cfg = make_env_cfg(
    rough=course not in ("flat_foundation", "flat_axes"), him_enabled=True
  )
  cfg.seed = seed
  cfg.scene.num_envs = count
  cfg.auto_reset = False
  cfg.curriculum = {}
  cfg.recorders = {}  # NativeEnv保存reset前的真实末帧，不再调用旧HIM recorder。
  cfg.sim.mujoco.cone = "elliptic"
  if not any(sensor.name == "terrain_scan" for sensor in cfg.scene.sensors):
    cfg.scene.sensors = (
      *cfg.scene.sensors,
      RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
      ),
    )
  cfg.observations["critic"].terms["privileged_terrain"] = ObservationTermCfg(
    func=rewards.privileged_terrain
  )
  cfg.events["reset_base"].func = reset_route
  cfg.events["reset_base"].params = {"pose_range": {}, "velocity_range": {}}
  old_command = cfg.commands["twist"]
  cfg.commands["twist"] = CourseCommandCfg(
    course=course,
    entity_name="robot",
    ranges=old_command.ranges,
    resampling_time_range=(27.0, 27.0) if course == "general" else (11.0, 11.0),
    heading_command=False,
    max_level=0 if max_level is None else max_level,
    allow_promotion=allow_promotion,
  )
  if course in SPECIALTIES:
    from support.black.obstacle_geometry import GROUPS, SpecialtyTerrainCfg

    cap = len(GROUPS[course]) - 1 if max_level is None else max_level
    if not 0 <= cap < len(GROUPS[course]):
      raise ValueError("专项max_level越界")
    cfg.commands["twist"].max_level = cap
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
      size=(30.0, 6.0),
      num_rows=len(GROUPS[course]),
      num_cols=16,
      border_width=5.0,
      curriculum=True,
      sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.25),
        course: SpecialtyTerrainCfg(task=course, proportion=0.75),
      },
    )
    cfg.scene.terrain.max_init_terrain_level = 0 if allow_promotion else cap
    cfg.episode_length_s = 120.0
    cfg.terminations["route_complete"] = TerminationTermCfg(func=route_finished)
    cfg.terminations["route_failed"] = TerminationTermCfg(func=route_failed)
    cfg.rewards["base_height"] = RewardTermCfg(
      func=rewards.support_height, weight=-10.0
    )
  elif course == "general":
    from mjlab.tasks.velocity.config.black_rough.terrain_cfgs import terrain_config

    cfg.scene.terrain.terrain_generator = terrain_config(seed)
    generator = cfg.scene.terrain.terrain_generator
    previous = sum(
      term.proportion for name, term in generator.sub_terrains.items() if name != "flat"
    )
    for name, term in generator.sub_terrains.items():
      term.proportion = 0.25 if name == "flat" else 0.75 * term.proportion / previous
    cfg.scene.terrain.max_init_terrain_level = 4 if max_level is None else max_level
    cfg.commands["twist"].max_level = cfg.scene.terrain.max_init_terrain_level
    cfg.episode_length_s = 27.0
  wheel_bodies = SceneEntityCfg(
    "robot", body_names=WOLF_WHEEL_BODY_NAMES, preserve_order=True
  )
  cfg.rewards["rolling_contact"] = RewardTermCfg(
    func=rewards.rolling_cost,
    weight=-0.05,
    params={"sensor_name": WOLF_WHEEL_SLIP_CONTACT_SENSOR, "asset_cfg": wheel_bodies},
  )
  cfg.rewards["lateral_contact"] = RewardTermCfg(
    func=rewards.rolling_cost,
    weight=-0.05,
    params={
      "sensor_name": WOLF_WHEEL_SLIP_CONTACT_SENSOR,
      "asset_cfg": wheel_bodies,
      "lateral": True,
    },
  )
  cfg.rewards["stop_body"] = RewardTermCfg(func=rewards.stop_body, weight=-2.0)
  cfg.rewards["stop_wheels"] = RewardTermCfg(
    func=rewards.stop_wheels,
    weight=-0.01,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=WOLF_WHEEL_JOINT_NAMES, preserve_order=True
      )
    },
  )
  legs = tuple(
    joint for leg in WOLF_LEG_ORDER for joint in WOLF_LEG_JOINT_NAMES[leg][:3]
  )
  cfg.rewards["stop_legs"] = RewardTermCfg(
    func=rewards.stop_legs,
    weight=-0.005,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=legs, preserve_order=True)
    },
  )
  # 不强迫轮足在障碍上回到标称腿姿；仅保留原hip弱约束及动作平滑。
  cfg.rewards.pop("stand_still", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  return cfg
