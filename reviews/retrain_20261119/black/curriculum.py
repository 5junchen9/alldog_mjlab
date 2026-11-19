"""从平地基础、纯轴链到综合地形和六专项的课程。"""

import copy
from dataclasses import fields

import torch

COURSES = ("flat_foundation", "flat_axes", "general", "specialty")
SETTLE_STEPS = 50
EVALUATION_SEEDS = (191301, 191401, 191501)
RESERVED_SEEDS = tuple(seed + batch for seed in EVALUATION_SEEDS for batch in range(3))


def training_seed(seed):
  if seed in RESERVED_SEEDS:
    raise ValueError("训练seed不能使用固定评估面板的seed")


def strict_parameters(original):
  parameters = copy.deepcopy(original)
  parameters.update(
    settle_steps=SETTLE_STEPS,
    finish_max_linear_speed=0.08,
    finish_max_angular_speed=0.12,
    finish_max_tilt_radians=0.35,
  )
  return parameters


def strict_passage_book(state):
  """为本环境安装严格票据；旧模块的SUCCESS集合保持不动。"""
  from support.black.passage_course import PassageCourse

  if getattr(state.passage_course, "v2_strict", False):
    return

  class StrictPassageCourse(PassageCourse):
    v2_strict = True

    def vote(self, lane, stamp, level, angle, direction, outcome, proof=None):
      if outcome == "pass_timeout_no_stop":
        outcome = "timeout_no_pass"
      return super().vote(lane, stamp, level, angle, direction, outcome, proof)

  if state.passage_course.active or state.passage_course.births:
    raise RuntimeError("v2严格课程必须在首条路线出生之前接入")
  state.passage_course = StrictPassageCourse(state.command.cfg.passage_course_policy)


def flat_commands(age, choice, course):
  """每条episode自己的时钟；摔倒后的新episode从静止段重新开始。"""
  targets = torch.zeros(len(age), 3, device=age.device)
  if course == "flat_foundation":
    table = torch.tensor(
      [
        [0.3, 0, 0],
        [-0.3, 0, 0],
        [0, 0.3, 0],
        [0, -0.3, 0],
        [0, 0, 0.5],
        [0, 0, -0.5],
        [0, 0, 0],
        [0, 0, 0],
      ],
      device=age.device,
    )
    phase = age % 500
    targets[:] = table[(choice + age // 500) % len(table)]
    targets[(phase < 100) | (phase >= 400)] = 0
  elif course == "specialty":
    # 专项25%平地回放：50%保留低速纯轴起停反转 + 50%中速/复合转向全向，防灾难性遗忘。
    phase = age % 900
    slot = (choice + age // 900) % 24
    pure = slot < 12
    if pure.any():
      idx = pure.nonzero().flatten()
      cell = slot[idx]
      axis = cell // 4
      magnitude = torch.where(cell % 4 < 2, 0.10, 0.25)
      sign = torch.where(cell % 2 == 0, -1.0, 1.0)
      targets[idx, axis] = magnitude * sign
    if (~pure).any():
      idx = (~pure).nonzero().flatten()
      mixed_table = torch.tensor(
        [
          [1.0, 0.0, 0.0],
          [-1.0, 0.0, 0.0],
          [0.0, 0.8, 0.0],
          [0.0, -0.8, 0.0],
          [0.7, 0.0, 0.45],
          [0.7, 0.0, -0.45],
          [-0.4, 0.0, 0.4],
          [-0.4, 0.0, -0.4],
          [0.35, 0.25, 0.35],
          [0.35, -0.25, -0.35],
          [0.0, 0.0, 0.5],
          [0.0, 0.0, 0.0],
        ],
        device=age.device,
      )
      targets[idx] = mixed_table[slot[idx] - 12]
    targets[(phase >= 350) & (phase < 600)] *= -1
    targets[(phase < 100) | (phase >= 600)] = 0
  else:
    phase = age % 900
    cell = (choice + age // 900) % 12
    axis = cell // 4
    magnitude = torch.where(cell % 4 < 2, 0.10, 0.25)
    sign = torch.where(cell % 2 == 0, -1.0, 1.0)
    targets[torch.arange(len(age), device=age.device), axis] = magnitude * sign
    targets[(phase >= 350) & (phase < 600)] *= -1
    targets[(phase < 100) | (phase >= 600)] = 0
  return targets


def _configure_specialty(cfg, course="flat_foundation"):
  """构造环境前修改该cfg；不修改导入的历史模块或全局参数。"""
  from support.black.specialty_composition import CompositionCommand
  from support.black.specialty_env import (
    SpecialtyCommand,
    SpecialtyCommandCfg,
    specialty_state,
  )

  if course != "specialty":
    raise ValueError("unknown v2 training course")
  original = cfg.commands["twist"]
  parameters = strict_parameters(original.course_parameters)

  class TrainingCommand(SpecialtyCommand):
    def __init__(self, command_cfg, env):
      super().__init__(command_cfg, env)
      self.training_choice = torch.arange(self.num_envs, device=self.device)
      self.training_shadow = self.vel_command_b.clone()

    def _resample_command(self, env_ids):
      # v2平地命令由publish产生，跳过只支持前1024条的历史覆盖增样器。
      # 上游原生命令时钟和楼梯路径速度仍照常更新。
      CompositionCommand._resample_command(self, env_ids)
      rough = env_ids[~self.flat_mask[env_ids]]
      self.target_command[rough] = 0
      self.target_command[rough, 0] = torch.empty(
        len(rough), device=self.device
      ).uniform_(0.15, 0.35)

    def publish(self):
      self.training_shadow.copy_(self.vel_command_b)
      ids = self.flat_mask.nonzero().flatten()
      self.vel_command_b[ids] = flat_commands(
        self._env.episode_length_buf[ids], self.training_choice[ids], course
      )

    def compute(self, dt, env_ids=None):
      self.vel_command_b.copy_(self.training_shadow)
      result = super().compute(dt, env_ids)
      self.publish()
      return result

    def reset(self, env_ids=None):
      self.vel_command_b.copy_(self.training_shadow)
      result = super().reset(env_ids)
      ids = (
        torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
      )
      self.training_choice[ids] += torch.randint(1, 13, (len(ids),), device=self.device)
      self.publish()
      return result

    def sample_route_entries(self, env_ids):
      state = specialty_state(self._env)
      if state.passage_course is not None:
        strict_passage_book(state)
      return super().sample_route_entries(env_ids)

  class TrainingCommandCfg(SpecialtyCommandCfg):
    def build(self, env):
      return TrainingCommand(self, env)

  values = {
    field.name: copy.deepcopy(getattr(original, field.name))
    for field in fields(original)
  }
  values["course_parameters"] = parameters
  cfg.commands["twist"] = TrainingCommandCfg(**values)
  old_course = cfg.curriculum["specialty_course"].func

  def update(env, env_ids):
    state = specialty_state(env)
    if state.passage_course is not None:
      strict_passage_book(state)
    return old_course(env, env_ids)

  cfg.curriculum["specialty_course"].func = update
  return dict(
    course=course,
    flat_fraction=0.25,
    strict_settle_steps=SETTLE_STEPS,
    timeout_is_success=False,
  )


def configure(cfg, course="flat_foundation"):
  """平地→综合→专项；综合保留原生低速/全向/高速分布。"""
  from support.black.specialty_composition import (
    CompositionCommand,
    CompositionCommandCfg,
  )

  if course not in COURSES:
    raise ValueError("unknown training course")
  if course == "specialty":
    return _configure_specialty(cfg, course)
  if course == "general":
    original = cfg.curriculum["rough_levels"].func

    def general_update(env, ids):
      from mjlab.tasks.velocity.config.black_rough.mdp import state

      st = state(env)
      if not hasattr(env, "general_progress"):
        env.general_progress = GeneralProgress()
      selected = torch.arange(env.num_envs, device=env.device) if ids is None else ids
      command = env.command_manager.get_term("twist")
      terrain = env.scene.terrain
      active = selected[
        (~command.flat_mask[selected]) & (env.episode_length_buf[selected] > 0)
      ]
      # Only real episode closure votes. A cold/administrative reset cannot promote.
      closed = env.termination_manager.terminated | env.termination_manager.time_outs
      active = active[closed[active]]
      rows = (
        torch.stack(
          (
            terrain.terrain_types[active],
            terrain.terrain_levels[active],
            (st.traversal[active] & ~st.failed[active]).long(),
          ),
          -1,
        )
        .cpu()
        .tolist()
      )
      for kind, level, passed in rows:
        env.general_progress.observe(kind, level, bool(passed), st.type_ceilings)
      st.maximum_level = max(st.type_ceilings)
      return original(env, ids)

    cfg.curriculum["rough_levels"].func = general_update
    return dict(course=course, flat_fraction=0.25)

  class FlatCommand(CompositionCommand):
    def __init__(self, command_cfg, env):
      super().__init__(command_cfg, env)
      self.flat_mask.fill_(True)
      terrain = env.scene.terrain
      terrain.terrain_types.zero_()
      terrain.terrain_levels.zero_()
      terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels, terrain.terrain_types
      ]
      self.training_choice = torch.arange(self.num_envs, device=self.device)
      self.training_shadow = self.vel_command_b.clone()

    def publish(self):
      self.training_shadow.copy_(self.vel_command_b)
      self.vel_command_b[:] = flat_commands(
        self._env.episode_length_buf, self.training_choice, course
      )

    def compute(self, dt, env_ids=None):
      self.vel_command_b.copy_(self.training_shadow)
      result = super().compute(dt, env_ids)
      self.publish()
      return result

    def reset(self, env_ids=None):
      self.vel_command_b.copy_(self.training_shadow)
      result = super().reset(env_ids)
      ids = (
        torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
      )
      self.training_choice[ids] += torch.randint(1, 13, (len(ids),), device=self.device)
      self.publish()
      return result

  class FlatCommandCfg(CompositionCommandCfg):
    def build(self, env):
      return FlatCommand(self, env)

  original = cfg.commands["twist"]
  cfg.commands["twist"] = FlatCommandCfg(
    **{
      field.name: copy.deepcopy(getattr(original, field.name))
      for field in fields(original)
    }
  )
  cfg.curriculum = {}
  return dict(course=course, flat_fraction=1.0)


class GeneralProgress:
  """每个地形独立解锁相邻一档；晋级训练与最终资格分开。"""

  def __init__(self):
    self.windows = {}

  def observe(self, kind, level, passed, ceilings):
    if kind == 0 or level != ceilings[kind] or level >= 4:
      return
    history = self.windows.setdefault((kind, level), [])
    history.append(passed)
    del history[:-64]
    if len(history) >= 32 and sum(history) / len(history) >= 0.90:
      ceilings[kind] += 1


def capture_curriculum(env):
  """只持久化已结算课程证据；物理、相机、GRU和在途通关状态不续接。"""
  plant = env.plant
  saved = dict(course=env.course, tick=plant.common_step_counter)
  if env.course == "general":
    from mjlab.tasks.velocity.config.black_rough.mdp import state

    st = state(plant)
    saved.update(
      ceilings=list(st.type_ceilings),
      windows=copy.deepcopy(
        getattr(plant, "general_progress", GeneralProgress()).windows
      ),
      histories=[list(rows) for rows in st.histories],
      levels=plant.scene.terrain.terrain_levels.cpu().clone(),
    )
  elif env.course == "specialty":
    from support.black.specialty_env import specialty_state

    st = specialty_state(plant)
    saved.update(
      height=st.height_level,
      angle=st.angle_stage,
      history=copy.deepcopy(st.history),
      promotions=copy.deepcopy(st.promotions),
      grid=None if st.grid is None else st.grid.state_dict(),
      passage=None if st.passage_course is None else st.passage_course.state_dict(),
    )
  return saved


def restore_curriculum(env, saved):
  if saved["course"] != env.course:
    raise ValueError("课程恢复阶段不一致")
  plant = env.plant
  if env.course == "general":
    from collections import deque

    from mjlab.tasks.velocity.config.black_rough.mdp import state

    st = state(plant)
    st.type_ceilings = list(saved["ceilings"])
    st.maximum_level = max(st.type_ceilings)
    st.histories = [deque(rows, maxlen=50) for rows in saved["histories"]]
    plant.general_progress = GeneralProgress()
    plant.general_progress.windows = copy.deepcopy(saved["windows"])
    plant.scene.terrain.terrain_levels.copy_(saved["levels"].to(env.device))
  elif env.course == "specialty":
    from support.black.grid_course import GridCurriculum
    from support.black.specialty_env import specialty_state

    st = specialty_state(plant)
    st.height_level, st.angle_stage = saved["height"], saved["angle"]
    st.history, st.promotions = (
      copy.deepcopy(saved["history"]),
      copy.deepcopy(saved["promotions"]),
    )
    if saved["grid"] is not None:
      st.grid = GridCurriculum.from_state(saved["grid"])
      st.history = st.grid.history
      # 不恢复旧物理路线，保留档位、已结算窗口、抽样配额及其随机序列。
      st.grid.assigned_birth.clear()
      st.routes.core.routes.clear()
      st.routes.core.last_birth.clear()
      st.routes.sync(torch.arange(env.count, device=env.device))
    if saved["passage"] is not None:
      st.passage_course = type(st.passage_course).restore(
        saved["passage"],
        st.command.cfg.passage_course_policy,
        cold_reset=True,
        explicit_candidate=True,
      )


def limit_student_specialty(cfg, max_level):
  """学生只采样当前教师已通过的连续难度范围和入场角。"""
  from support.black.obstacle_geometry import GROUPS
  from support.black.specialty_env import specialty_state

  from black.eval_protocol import HEIGHTS_CM

  command = cfg.commands["twist"]
  task = command.task
  if not 0 <= max_level < len(GROUPS[task]):
    raise ValueError("学生专项难度上限无效")
  if task == "stairs":
    if max_level >= len(HEIGHTS_CM):
      raise ValueError("楼梯范围超出当前通过协议")
    command.passage_course_policy["maximum_cm"] = HEIGHTS_CM[max_level]
    return
  original = cfg.curriculum["specialty_course"].func

  def update(env, ids):
    result = original(env, ids)
    state = specialty_state(env)
    state.height_level = min(state.height_level, max_level)
    state.angle_stage = min(state.angle_stage, 1)  # 固定面板覆盖0、±15°。
    terrain = env.scene.terrain
    terrain.terrain_levels.clamp_(max=max_level)
    terrain.env_origins[:] = terrain.terrain_origins[
      terrain.terrain_levels, terrain.terrain_types
    ]
    result.update(height_level=state.height_level, entry_angle_stage=state.angle_stage)
    return result

  cfg.curriculum["specialty_course"].func = update
