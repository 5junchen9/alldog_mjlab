"""构造前固定路线与命令，命令在原生history计算前发布。"""

import math
from dataclasses import dataclass, fields

import torch

from mjlab.envs import mdp
from wolf.curriculum import SPECIALTIES, CourseCommand, CourseCommandCfg, reset_route


class EvaluationCommand(CourseCommand):
  def _resample_command(self, env_ids):
    self.target[env_ids] = torch.tensor(
      [self.cfg.rows[int(i)]["target"] for i in env_ids], device=self.device
    )


@dataclass(kw_only=True)
class EvaluationCommandCfg(CourseCommandCfg):
  rows: list

  def build(self, env):
    return EvaluationCommand(self, env)


def fixed_reset(env, env_ids, pose_range, velocity_range):
  command = env.command_manager.get_term("twist")
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  rows = command.cfg.rows
  terrain = env.scene.terrain
  if command.course in SPECIALTIES or command.course == "general":
    terrain.terrain_levels[ids] = torch.tensor(
      [rows[int(i)]["level"] for i in ids], device=env.device
    )
    terrain.terrain_types[ids] = torch.tensor(
      [rows[int(i)].get("domain", 1) for i in ids], device=env.device
    )
    terrain.env_origins[ids] = terrain.terrain_origins[
      terrain.terrain_levels[ids], terrain.terrain_types[ids]
    ]
    if command.course in SPECIALTIES:
      command.flat_mask[ids] = False
  reset_route(env, ids, pose_range, velocity_range)
  for row_number, lane in enumerate(ids.tolist()):
    row = rows[lane]
    direction = row["direction"]
    command.direction[lane] = direction
    offset = (
      float(command.lengths[row["level"]])
      if command.course in SPECIALTIES and direction < 0
      else 0.0
    )
    yaw = row.get("orientation", math.pi if direction < 0 else 0.0)
    selected = ids[row_number : row_number + 1]
    mdp.reset_root_state_uniform(
      env, selected, {"x": (offset, offset), "yaw": (yaw, yaw)}, {}
    )


def snapshot(env):
  robot = env.scene["robot"].data
  command = env.command_manager.get_term("twist")
  sensor = env.scene["wheel_slip_contact"].data
  return {
    "velocity": robot.root_link_lin_vel_b.clone(),
    "angular": robot.root_link_ang_vel_b.clone(),
    "tilt": torch.acos((-robot.projected_gravity_b[:, 2]).clamp(-1, 1)),
    "contacts": ((sensor.found > 0) & (sensor.force[..., 0].abs() > 1.0)).clone(),
    "contact_points": (sensor.pos - env.scene.env_origins[:, None, :]).clone(),
    "root": (robot.root_link_pos_w - env.scene.env_origins).clone(),
    "command": command.command.clone(),
    "route_passed": command.passed.clone(),
    "route_failed": command.failed.clone(),
    "route_settled": command.settled.clone(),
    "route_entered": command.entered.clone(),
    "route_contact_steps": command.contact_steps.clone(),
    "action": env.action_manager.action.clone(),
    "raw_action": torch.cat(
      [
        env.action_manager.get_term(name).raw_action
        for name in env.action_manager.active_terms
      ],
      -1,
    ).clone(),
    "processed_targets": torch.cat(
      [
        env.action_manager.get_term(name)._processed_actions
        for name in env.action_manager.active_terms
      ],
      -1,
    ).clone(),
    "native_failure": torch.stack(
      [
        env.termination_manager.get_term(name)
        for name in env.termination_manager.active_terms
        if name not in ("route_complete", "time_out")
      ]
    ).any(0),
    "timeout": env.termination_manager.time_outs.clone(),
    "lower_floor": (
      command.sparse_catalog.read(env.sim.data, env.num_envs)["lower_floor"].any(-1)
      if command.sparse_catalog is not None
      else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ),
  }


class EvaluationAdapter:
  def __init__(self, rows):
    self.rows = rows
    self.env = None
    self.closed = None

  def configure(self, cfg):
    old = cfg.commands["twist"]
    parameters = {field.name: getattr(old, field.name) for field in fields(old)}
    cfg.commands["twist"] = EvaluationCommandCfg(**parameters, rows=self.rows)
    cfg.commands["twist"].allow_promotion = False
    cfg.events["reset_base"].func = fixed_reset
    # 独立静态地图与命令评价，不施加训练中的随机外部推搡。
    cfg.events = {
      key: value
      for key, value in cfg.events.items()
      if value.mode not in ("interval", "step")
    }

  def attach(self, env):
    self.env = env

  def begin(self, native):
    self.before = snapshot(self.env.plant)

  def capture_terminal(self):
    self.closed = snapshot(self.env.plant)

  def finish(self, native, done):
    pass

  def detach(self):
    self.env = None
