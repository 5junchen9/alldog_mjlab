"""平地起步、通用地形和六专项；专项与综合均含独立课程与25%平地回放。"""

import copy
import math
from dataclasses import dataclass

import torch
from black.curriculum import GeneralProgress

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
from mjlab.tasks.velocity import mdp as velocity_mdp
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


def capture_curriculum(env):
  command = env.plant.command_manager.get_term("twist")
  return dict(
    course=env.course,
    tick=env.plant.common_step_counter,
    unlocked_level=command.unlocked_level,
    promotion_successes=command.promotion_successes,
    promotion_results=list(command.promotion_results),
    general_ceilings=list(command.general_ceilings),
    general_windows=copy.deepcopy(command.general_progress.windows),
  )


def restore_curriculum(env, saved):
  if saved["course"] != env.course:
    raise ValueError("Wolf课程恢复阶段不一致")
  command = env.plant.command_manager.get_term("twist")
  for key in (
    "unlocked_level",
    "promotion_successes",
    "promotion_results",
    "general_ceilings",
  ):
    setattr(command, key, copy.deepcopy(saved[key]))
  command.general_progress.windows = copy.deepcopy(saved["general_windows"])


def flat_command(tick, target, general=False):
  """先静止1秒，再起动、停、反向、停；命令不经过旧0.2m/s死区。"""
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
    self.specialty_speed = torch.full((self.num_envs,), 0.2, device=self.device)
    self.flat_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self.direction = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
    self.settled = torch.zeros_like(self.direction)
    self.passed = torch.zeros_like(self.flat_mask)
    self.failed = torch.zeros_like(self.flat_mask)
    self.entered = torch.zeros_like(self.flat_mask)
    self.contact_steps = torch.zeros_like(self.direction)
    self.last_root_progress = torch.zeros(self.num_envs, device=self.device)
    self.last_contact_count = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.general_left_platform = torch.zeros_like(self.flat_mask)
    self.general_escaped = torch.zeros_like(self.flat_mask)
    self.general_demand = torch.zeros(self.num_envs, device=self.device)
    self.general_progress_m = torch.zeros_like(self.general_demand)
    self.general_motion_steps = torch.zeros_like(self.direction)
    self.general_tracking_steps = torch.zeros_like(self.direction)
    self.last_tick = -1
    self.lengths = self.widths = None
    self.cases = []
    self.unlocked_level = 0
    self.promotion_successes = 0
    self.promotion_results = []
    self.general_ceilings = [0, 1, 1, 1, 1, 1, 1]
    self.general_progress = GeneralProgress()
    self.sparse_catalog = None
    self.treads = []
    self.ordered = torch.zeros_like(self.direction)
    self.touched = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
    if self.course == "general":
      terrain = env.scene.terrain
      self.flat_mask = terrain.terrain_types == 0
      if cfg.allow_promotion:
        rough = ~self.flat_mask
        terrain.terrain_levels[rough] = 1
        terrain.env_origins[rough] = terrain.terrain_origins[
          terrain.terrain_levels[rough], terrain.terrain_types[rough]
        ]
    elif self.course in SPECIALTIES:
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
    n = len(env_ids)
    device = self.device
    self.target[env_ids] = 0
    course = self.cfg.course
    if course == "flat_foundation":
      # 基础阶段先学低速前后起停；更高速度和复合命令留给后续阶段。
      signs = torch.randint(2, (n,), device=device) * 2 - 1
      self.target[env_ids, 0] = signs * torch.empty(n, device=device).uniform_(0.2, 0.3)
      self.target[env_ids[torch.rand(n, device=device) < 0.25]] = 0
      return
    # flat_axes / general / specialty平地回放：同时覆盖单轴与复合转向(vx+wz/全向)。
    draw = torch.rand(n, device=device)
    stand = draw < 0.20
    mixed = (draw >= 0.20) & (draw < (0.55 if course == "general" else 0.50))
    single = ~(stand | mixed)
    if single.any():
      idx = env_ids[single]
      m = len(idx)
      axis = torch.randint(3, (m,), device=device)
      signs = torch.randint(2, (m,), device=device) * 2 - 1
      low_band = torch.rand(m, device=device) < 0.5
      if course == "general" or course in SPECIALTIES:
        vx_mag = torch.where(
          low_band,
          torch.empty(m, device=device).uniform_(0.25, 0.40),
          torch.empty(m, device=device).uniform_(0.50, 1.20),
        )
        vy_mag = torch.where(
          low_band,
          torch.empty(m, device=device).uniform_(0.20, 0.35),
          torch.empty(m, device=device).uniform_(0.35, 0.60),
        )
        wz_mag = torch.where(
          low_band,
          torch.empty(m, device=device).uniform_(0.30, 0.50),
          torch.empty(m, device=device).uniform_(0.50, 1.20),
        )
      else:
        vx_mag = torch.where(
          low_band,
          torch.empty(m, device=device).uniform_(0.20, 0.30),
          torch.empty(m, device=device).uniform_(0.40, 0.80),
        )
        vy_mag = torch.where(
          low_band,
          torch.empty(m, device=device).uniform_(0.20, 0.30),
          torch.empty(m, device=device).uniform_(0.30, 0.50),
        )
        wz_mag = torch.where(
          low_band,
          torch.empty(m, device=device).uniform_(0.30, 0.50),
          torch.empty(m, device=device).uniform_(0.50, 1.00),
        )
      mags = torch.where(axis == 0, vx_mag, torch.where(axis == 1, vy_mag, wz_mag))
      self.target[idx, axis] = signs * mags
    if mixed.any():
      idx = env_ids[mixed]
      m = len(idx)
      vx_hi = 1.20 if course == "general" else 0.80
      wz_hi = 1.00 if course == "general" else 0.80
      vx_sign = torch.randint(2, (m,), device=device) * 2 - 1
      wz_sign = torch.randint(2, (m,), device=device) * 2 - 1
      self.target[idx, 0] = vx_sign * torch.empty(m, device=device).uniform_(
        0.20, vx_hi
      )
      self.target[idx, 2] = wz_sign * torch.empty(m, device=device).uniform_(
        0.20, wz_hi
      )
      omni = torch.rand(m, device=device) < 0.30
      if omni.any():
        vy_sign = torch.randint(2, (int(omni.sum()),), device=device) * 2 - 1
        self.target[idx[omni], 1] = vy_sign * torch.empty(
          int(omni.sum()), device=device
        ).uniform_(0.15, 0.40)
    if course in SPECIALTIES:
      rough = env_ids[~self.flat_mask[env_ids]]
      if len(rough):
        self.specialty_speed[rough] = torch.empty(len(rough), device=device).uniform_(
          0.20, 0.50
        )

  def reset(self, env_ids):
    result = super().reset(env_ids)
    self.settled[env_ids] = 0
    self.passed[env_ids] = False
    self.failed[env_ids] = False
    self.entered[env_ids] = False
    self.contact_steps[env_ids] = 0
    self.last_root_progress[env_ids] = 0.0
    self.last_contact_count[env_ids] = 0
    self.general_left_platform[env_ids] = False
    self.general_escaped[env_ids] = False
    self.general_demand[env_ids] = 0.0
    self.general_progress_m[env_ids] = 0.0
    self.general_motion_steps[env_ids] = 0
    self.general_tracking_steps[env_ids] = 0
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
    if self.course == "general":
      rough = ids[~self.flat_mask[ids]]
      if len(rough):
        offset = (
          (
            self.robot.data.root_link_pos_w[rough, :2]
            - self._env.scene.env_origins[rough, :2]
          )
          .abs()
          .amax(-1)
        )
        self.general_left_platform[rough] |= offset > 1.0
        self.general_escaped[rough] |= offset > 4.0
        cmd = self.vel_command_b[rough, :2]
        speed = cmd.norm(dim=-1)
        moving = speed > 0.05
        actual = self.robot.data.root_link_lin_vel_b[rough, :2]
        projected = (actual * cmd).sum(-1) / speed.clamp_min(1e-8)
        self.general_demand[rough] += torch.where(
          moving, speed * self._env.step_dt, 0.0
        )
        self.general_progress_m[rough] += torch.where(
          moving, projected.clamp(min=0.0) * self._env.step_dt, 0.0
        )
        self.general_motion_steps[rough] += moving.long()
        self.general_tracking_steps[rough] += (
          moving & (projected >= 0.5 * speed)
        ).long()
      return
    if self.course not in SPECIALTIES:
      return
    ids = ids[~self.flat_mask[ids]]
    if not len(ids):
      return
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
    speed = torch.where(
      progress < self.lengths[level] - 0.15, self.specialty_speed[ids], 0.0
    )
    speed = torch.where(self._env.episode_length_buf[ids] < 50, 0.0, speed)
    # 轮足纠偏采用前向滚动+偏航转向，避免发vy强迫轮子横向硬蹭。
    self.vel_command_b[ids, 0] = speed * error.cos().clamp(min=0.0)
    self.vel_command_b[ids, 1] = 0.0
    self.vel_command_b[ids, 2] = torch.where(
      speed > 0, (1.5 * error).clamp(-0.6, 0.6), 0.0
    )

  def evidence(self):
    """每控制步一次，在reset前读取材料接触点；底板不能代替桥面/桩顶。"""
    if (
      self.course not in SPECIALTIES or self.last_tick == self._env.common_step_counter
    ):
      return
    self.last_tick = self._env.common_step_counter
    env = self._env
    ids = (~self.flat_mask).nonzero().flatten()
    if not len(ids):
      return
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
    self.last_root_progress[ids] = root_progress
    self.last_contact_count[ids] = contacts.sum(-1)
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
  if command.course == "general" and command.cfg.allow_promotion:
    terrain = env.scene.terrain
    rough_ids = ids[~command.flat_mask[ids]]
    if len(rough_ids):
      attempted = rough_ids[env.episode_length_buf[rough_ids] > 0]
      if len(attempted):
        closed = attempted[
          env.termination_manager.terminated[attempted]
          | env.termination_manager.time_outs[attempted]
        ]
        if len(closed):
          passed = (
            (~env.termination_manager.terminated[closed])
            & command.general_left_platform[closed]
            & (~command.general_escaped[closed])
            & (command.general_demand[closed] >= 1.0)
            & (
              command.general_progress_m[closed] >= 0.8 * command.general_demand[closed]
            )
            & (
              command.general_tracking_steps[closed]
              >= 0.90 * command.general_motion_steps[closed]
            )
          )
          rows = (
            torch.stack(
              (
                terrain.terrain_types[closed],
                terrain.terrain_levels[closed],
                passed.long(),
              ),
              -1,
            )
            .cpu()
            .tolist()
          )
          cap = max(1, min(4, int(command.cfg.max_level or 4)))
          for kind, level, ok in rows:
            command.general_progress.observe(
              kind, level, bool(ok), command.general_ceilings
            )
            command.general_ceilings[kind] = min(command.general_ceilings[kind], cap)
      kinds = terrain.terrain_types[rough_ids]
      ceilings = torch.tensor(
        command.general_ceilings, device=env.device, dtype=torch.long
      )[kinds].clamp(min=1)
      levels = ceilings.clone()
      replay = (ceilings > 1) & (torch.rand(len(levels), device=env.device) < 0.25)
      if replay.any():
        # 在已解锁的[1, ceiling]范围内回放，不退回纯平地level 0。
        span = (ceilings[replay] - 1).clamp(min=1)
        levels[replay] = (
          1 + (torch.rand(int(replay.sum()), device=env.device) * span).long()
        )
      terrain.terrain_levels[rough_ids] = levels
      terrain.env_origins[rough_ids] = terrain.terrain_origins[levels, kinds]
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
  # 同一步硬失败优先，不能同时拿通关奖励或晋级票。
  hard_failures = [
    env.termination_manager.get_term(name)
    for name in env.termination_manager.active_terms
    if name not in ("route_complete", "route_failed", "time_out")
  ]
  if hard_failures:
    command.failed |= torch.stack(hard_failures).any(0)
  return command.passed & ~command.failed


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
  # 全阶段补齐翻车倾角终止；general保留地形越界终止，仅专项使用自有廊道边界。
  cfg.terminations["bad_orientation"] = TerminationTermCfg(
    func=velocity_mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  # 替换为双尺度速度跟踪核，解决低速0.2m/s下静止白嫖85%满分的问题。
  cfg.rewards["tracking_lin_vel_x"] = RewardTermCfg(
    func=rewards.track_linear_velocity_x,
    weight=1.0,
    params={"command_name": "twist", "sigma": 0.25, "fine_sigma": 0.02},
  )
  cfg.rewards["tracking_lin_vel_y"] = RewardTermCfg(
    func=rewards.track_linear_velocity_y,
    weight=1.0,
    params={"command_name": "twist", "sigma": 0.25, "fine_sigma": 0.02},
  )
  cfg.rewards["tracking_ang_vel"] = RewardTermCfg(
    func=rewards.track_angular_velocity_z,
    weight=0.75,
    params={"command_name": "twist", "sigma": 0.25, "fine_sigma": 0.02},
  )
  if course in SPECIALTIES:
    rewards.verify_completion_bound(cfg)
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
    cfg.terminations.pop("out_of_terrain_bounds", None)
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
    cap = 4 if max_level is None else max_level
    cfg.scene.terrain.max_init_terrain_level = 1 if allow_promotion else cap
    cfg.commands["twist"].max_level = cap
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
  if course in SPECIALTIES:
    cfg.rewards["route_completion_bonus"] = RewardTermCfg(
      func=rewards.RouteCompletionBonus, weight=1.0
    )
    cfg.rewards["route_progress_shaping"] = RewardTermCfg(
      func=rewards.RouteProgressShaping, weight=1.0
    )
    cfg.rewards["route_stall_penalty"] = RewardTermCfg(
      func=rewards.RouteStallPenalty, weight=-1.0
    )
  # 不强迫轮足在障碍上回到标称腿姿；仅保留原hip弱约束及动作平滑。
  cfg.rewards.pop("stand_still", None)
  return cfg
