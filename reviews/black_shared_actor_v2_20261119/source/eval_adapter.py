"""本次评估私有的命令/路线适配；原生奖励、终止和自动reset保留。"""

import copy
import math
from dataclasses import fields
from typing import Any

import torch
from curriculum import strict_parameters, strict_passage_book


class EvaluationAdapter:
  def __init__(self, rows):
    self.rows = copy.deepcopy(rows)
    kinds = {row["kind"] for row in self.rows}
    if len(kinds) != 1 or not kinds <= {"flat", "stairs"}:
      raise ValueError("一个评估批次必须全为flat或全为stairs")
    self.kind = self.rows[0]["kind"]
    self.env: Any = None
    self.armed: dict[str, Any] | None = None
    self.evidence: dict[str, Any] | None = None
    self.consumers = set()

  def configure(self, cfg):
    """必须在原生环境构造前调用，只替换本次cfg的私有实现。"""
    from specialty_env import SpecialtyCommand, SpecialtyCommandCfg, specialty_state

    adapter = self

    class EvaluationCommand(SpecialtyCommand):
      def __init__(self, command_cfg, env):
        super().__init__(command_cfg, env)
        if self.num_envs != len(adapter.rows):
          raise ValueError("评估roster与原生环境数不一致")
        self.flat_mask.fill_(adapter.kind == "flat")
        terrain = env.scene.terrain
        # 原生第0列为专用平地，其余6列使用同一阶梯几何配置。
        terrain.terrain_types.fill_(0 if adapter.kind == "flat" else 1)
        terrain.terrain_levels.zero_()
        terrain.env_origins[:] = terrain.terrain_origins[
          terrain.terrain_levels, terrain.terrain_types
        ]
        self.evaluation_shadow = self.vel_command_b.clone()

      def publish(self):
        if adapter.kind == "flat":
          from eval_protocol import command

          self.evaluation_shadow.copy_(self.vel_command_b)
          tick = int(self._env.common_step_counter)
          self.vel_command_b.copy_(
            torch.tensor(
              [command(row, tick) for row in adapter.rows],
              dtype=self.vel_command_b.dtype,
              device=self.device,
            )
          )

      def compute(self, dt, env_ids=None):
        if adapter.kind == "flat":
          self.vel_command_b.copy_(self.evaluation_shadow)
        result = super().compute(dt, env_ids)
        self.publish()
        return result

      def reset(self, env_ids=None):
        if adapter.kind == "flat":
          self.vel_command_b.copy_(self.evaluation_shadow)
        result = super().reset(env_ids)
        self.publish()
        return result

      def sample_route_entries(self, env_ids):
        adapter.assign(specialty_state(self._env), env_ids)

    class EvaluationCommandCfg(SpecialtyCommandCfg):
      def build(self, env):
        return EvaluationCommand(self, env)

    original = cfg.commands["twist"]
    original.course_parameters = strict_parameters(original.course_parameters)
    cfg.commands["twist"] = EvaluationCommandCfg(
      **{
        field.name: copy.deepcopy(getattr(original, field.name))
        for field in fields(original)
      }
    )

    def fixed_course(env, env_ids):
      state = specialty_state(env)
      strict_passage_book(state)
      ids = (
        torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
      )
      state.record(ids)
      adapter.assign(state, ids)
      return {"evaluation_fixed_roster": True}

    cfg.curriculum["specialty_course"].func = fixed_course
    original_reset = cfg.events["reset_base"].func

    def reset(env, env_ids, pose_range, velocity_range):
      if adapter.kind == "stairs":
        return original_reset(env, env_ids, pose_range, velocity_range)
      ids = (
        torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
      )
      for orientation in sorted({row["orientation"] for row in adapter.rows}):
        selected = [
          i for i in ids.tolist() if adapter.rows[i]["orientation"] == orientation
        ]
        if selected:
          ranges = copy.deepcopy(pose_range)
          lo, hi = ranges.get("yaw", (0.0, 0.0))
          ranges["yaw"] = (lo + orientation, hi + orientation)
          original_reset(
            env, torch.tensor(selected, device=env.device), ranges, velocity_range
          )

    cfg.events["reset_base"].func = reset
    for name in ("track_linear_velocity", "track_angular_velocity"):
      original_reward = cfg.rewards[name].func

      def witnessed(env, *args, _original=original_reward, _name=name, **kwargs):
        self.witness(env, _name)
        return _original(env, *args, **kwargs)

      cfg.rewards[name].func = witnessed

  def assign(self, state, ids):
    """原生reset前固定路线，仍让原reset_specialty写入物理初始状态。"""
    if self.kind == "flat":
      return
    from passage_course import HEIGHTS

    env = state.env
    strict_passage_book(state)
    stamp = int(env.common_step_counter)
    pending = [i for i in ids.tolist() if state.grid.assigned_birth.get(i) != stamp]
    if not pending:
      return
    index = torch.tensor(pending, dtype=torch.long, device=env.device)
    levels = [HEIGHTS.index(self.rows[i]["height_cm"]) for i in pending]
    directions = [self.rows[i]["direction"] for i in pending]
    angles = [self.rows[i]["angle_deg"] for i in pending]
    for i in pending:
      state.routes.core.routes.pop(i, None)
    state.routes.sync(index)
    terrain = env.scene.terrain
    level = torch.tensor(levels, device=env.device)
    direction = torch.tensor(directions, device=env.device)
    terrain.terrain_levels[index] = level
    terrain.env_origins[index] = terrain.terrain_origins[
      level, terrain.terrain_types[index]
    ]
    state.command.entry_angle[index] = (
      torch.tensor(angles, device=env.device) * math.pi / 180
    )
    state.command.route_direction[index] = direction
    state.command.route_sample_step[index] = stamp
    book = state.passage_course
    for i, height, angle, sign in zip(pending, levels, angles, directions, strict=True):
      if i in book.active:
        raise RuntimeError("上一条阶梯trial尚未原生闭合")
      book.active[i] = (stamp, height, angle, sign)
      book.births += 1
      book.assigned[height, angle] += 1
      state.grid.assigned_birth[i] = stamp
      state.grid.retained.add(f"{height}/{angle}/{sign}")
    highest = max(levels)
    state.grid.highest_sampled = max(state.grid.highest_sampled, highest)
    state.height_level = max(state.height_level, highest)
    state.passage_seen.on_assign(index, stamp, level, direction)

  def attach(self, env):
    """克隆原奖励计算后的状态，此时自动reset和外力事件均尚未执行。"""
    self.env = env
    self.original_reward_compute = env.plant.reward_manager.compute
    self.original_metrics_update = env.metrics.update
    self.safe_exit = torch.zeros(len(self.rows), dtype=torch.bool)

    def metrics_update(flags, *args, **kwargs):
      result = self.original_metrics_update(flags, *args, **kwargs)
      self.safe_exit = self.clone(flags[:, 4])
      return result

    def compute(dt):
      result = self.original_reward_compute(dt)
      self.capture()
      return result

    env.plant.reward_manager.compute = compute
    env.metrics.update = metrics_update

  def detach(self):
    if self.env is not None:
      self.env.plant.reward_manager.compute = self.original_reward_compute
      self.env.metrics.update = self.original_metrics_update
      self.env = None

  @staticmethod
  def clone(value):
    return value.detach().cpu().clone()

  def pre(self):
    env = self.env
    plant = env.plant
    robot = plant.scene["robot"]
    linear = plant.scene["robot/imu_lin_vel"].data
    angular = plant.scene["robot/imu_ang_vel"].data
    velocity = torch.stack((linear[:, 0], linear[:, 1], angular[:, 2]), -1)
    gravity = robot.data.projected_gravity_b
    tilt = torch.acos((-gravity[:, 2] / gravity.norm(dim=-1)).clamp(-1, 1))
    feet = plant.scene["feet_ground_contact"].data.found > 0
    count = feet.sum(-1)
    return dict(
      tick=env.frames.tick,
      episode=self.clone(env.frames.episode),
      command=self.clone(plant.command_manager.get_command("twist")),
      velocity=self.clone(velocity),
      tilt=self.clone(tilt),
      contacts=self.clone(count),
      support=self.clone(count >= 2),
      foot_contacts=self.clone(feet),
    )

  def begin(self, native):
    if self.armed is not None:
      raise RuntimeError("上一评估动作尚未完成")
    self.armed = self.pre()
    self.check_history(native)
    self.evidence = None
    self.consumers.clear()

  def check_history(self, native):
    current = native["history"].reshape(len(self.rows), 5, 45)[:, -1, 30:33]
    actual = self.env.plant.command_manager.get_command("twist")
    if not torch.equal(current, actual):
      raise RuntimeError("当前history与原生命令不同帧")
    if self.kind == "flat":
      from eval_protocol import command

      tick = int(self.env.plant.common_step_counter)
      expected = torch.tensor(
        [command(row, tick) for row in self.rows], device=actual.device
      )
      if not torch.equal(actual, expected):
        raise RuntimeError("当前命令偏离预登记时序")

  def witness(self, plant, name):
    if self.armed is None:
      raise RuntimeError("原生奖励读取了未登记动作的命令")
    if not torch.equal(
      self.clone(plant.command_manager.get_command("twist")), self.armed["command"]
    ):
      raise RuntimeError("原生奖励命令与动作时刻不一致")
    self.consumers.add(name)

  def capture(self):
    from passage_course import HEIGHTS
    from specialty_env import specialty_state

    assert self.armed is not None
    env = self.env
    plant = env.plant
    state: Any = specialty_state(plant)
    robot = plant.scene["robot"].data
    gravity = robot.projected_gravity_b
    seen = state.passage_seen
    action_term = plant.action_manager.get_term("joint_pos")
    delay = action_term.control_delay.lags
    selected = action_term.control_delay.history[
      torch.arange(len(self.rows), device=delay.device), delay
    ]
    self.evidence = dict(
      tick=self.armed["tick"],
      terminal_tick=int(plant.common_step_counter),
      closed_tick=int(plant.common_step_counter),
      episode=self.armed["episode"].clone(),
      terminated=self.clone(plant.reset_terminated),
      truncated=self.clone(plant.reset_time_outs),
      failed=self.clone(state.failed),
      strict_finished=self.clone(state.finished),
      ordered_treads=self.clone(seen.stage),
      landing=self.clone(seen.landing),
      pass_once=self.clone(seen.pass_once),
      pass_invalid=self.clone(seen.invalid),
      first_verified_pass_tick=self.clone(seen.first_verified_pass_tick),
      birth=self.clone(state.command.route_sample_step),
      height_cm=(
        torch.tensor(HEIGHTS)[self.clone(plant.scene.terrain.terrain_levels)]
        if self.kind == "stairs"
        else torch.zeros(len(self.rows), dtype=torch.long)
      ),
      direction=self.clone(state.command.route_direction),
      angle_deg=self.clone(
        torch.round(state.command.entry_angle * 180 / math.pi).long()
      ),
      settled_steps=self.clone(state.ledger.settled_steps),
      safe_exit=self.safe_exit.clone(),
      action=self.clone(plant.action_manager.action),
      raw_action=self.clone(action_term.raw_action),
      delayed_action=self.clone(selected),
      processed_target=self.clone(action_term._processed_actions),
      action_delay=self.clone(delay),
      post_linear_speed=self.clone(robot.root_link_lin_vel_w.norm(dim=-1)),
      post_angular_speed=self.clone(robot.root_link_ang_vel_w.norm(dim=-1)),
      post_tilt=self.clone(
        torch.acos((-gravity[:, 2] / gravity.norm(dim=-1)).clamp(-1, 1))
      ),
      post_contacts=self.clone(plant.scene["feet_ground_contact"].data.found > 0),
    )

  def finish(self, native, done):
    assert self.armed is not None
    expected = {"track_linear_velocity", "track_angular_velocity"}
    if self.evidence is None or self.consumers != expected:
      raise RuntimeError("缺少原生终止前证据或tracking命令见证")
    if self.evidence["terminal_tick"] != self.armed["tick"] + 1:
      raise RuntimeError("评估控制时刻不连续")
    flags = self.evidence["terminated"] | self.evidence["truncated"]
    if not torch.equal(flags, self.clone(done).bool()) or not torch.equal(
      self.clone(self.env.frames.episode), self.armed["episode"] + flags.long()
    ):
      raise RuntimeError("原生终止与episode重置不一致")
    self.check_history(native)
    self.evidence["reward_command_verified"] = True
    self.armed = None

  def post(self):
    if self.armed is not None or self.evidence is None:
      raise RuntimeError("尚无完整的评估动作证据")
    return self.evidence
