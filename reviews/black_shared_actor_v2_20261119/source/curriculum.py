"""v2私有课程：先全平地，再纯轴链，最后楼梯并保留25%平地。"""

import copy
from dataclasses import fields

import torch

COURSES = ("flat_foundation", "flat_axes", "stairs")
SETTLE_STEPS = 50
EVALUATION_SEEDS = (191301, 191401, 191501)
RESERVED_SEEDS = tuple(seed + batch for seed in EVALUATION_SEEDS for batch in (0, 1))


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
  from passage_course import PassageCourse

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


def configure(cfg, course="flat_foundation"):
  """构造环境前修改该cfg；不修改导入的历史模块或全局参数。"""
  from specialty_composition import CompositionCommand
  from specialty_env import SpecialtyCommand, SpecialtyCommandCfg, specialty_state

  if course not in COURSES:
    raise ValueError("unknown v2 training course")
  original = cfg.commands["twist"]
  parameters = strict_parameters(original.course_parameters)

  class TrainingCommand(SpecialtyCommand):
    def __init__(self, command_cfg, env):
      super().__init__(command_cfg, env)
      if course != "stairs":
        self.flat_mask.fill_(True)
        terrain = env.scene.terrain
        terrain.terrain_types.zero_()
        terrain.terrain_levels.zero_()
        terrain.env_origins[:] = terrain.terrain_origins[
          terrain.terrain_levels, terrain.terrain_types
        ]
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
      strict_passage_book(specialty_state(self._env))
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
    strict_passage_book(specialty_state(env))
    return old_course(env, env_ids)

  cfg.curriculum["specialty_course"].func = update
  return dict(
    course=course,
    flat_fraction=0.25 if course == "stairs" else 1.0,
    strict_settle_steps=SETTLE_STEPS,
    timeout_is_success=False,
  )
