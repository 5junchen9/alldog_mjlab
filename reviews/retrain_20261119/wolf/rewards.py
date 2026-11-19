"""轮胎材料接触点滑移、双尺度速度跟踪、停车与专项防陷阱奖励。"""

import math

import torch

from mjlab.tasks.velocity.config.wolf.mdp.common import wheel_slip_velocities

ROUTE_COMPLETION_BONUS = 7.5
ROUTE_PROGRESS_PER_M = 0.5
ROUTE_SETTLE_SHAPING_TOTAL = 1.0
STALL_WINDOW_STEPS = 100
STALL_GRACE_SECONDS = 3.0
STALL_DEMAND_FRACTION = 0.25
STALL_DEADBAND_M = 0.005
STALL_MAX_CREDIBLE_STEP_M = 0.25
WOLF_POSITIVE_TRACKING_TERMS = (
  "tracking_lin_vel_x",
  "tracking_lin_vel_y",
  "tracking_ang_vel",
)


def verify_completion_bound(cfg, gamma=0.99):
  positive = {
    name: term.weight
    for name, term in cfg.rewards.items()
    if term.weight > 0 and not name.startswith("route_")
  }
  expected = dict(tracking_lin_vel_x=1.0, tracking_lin_vel_y=1.0, tracking_ang_vel=0.75)
  if positive != expected:
    raise ValueError("Wolf通关补偿需要重新核对正奖励上界")
  dt = cfg.sim.mujoco.timestep * cfg.decimation
  tail = sum(positive.values()) * dt / (1.0 - gamma)
  if not math.isclose(dt, 0.02) or ROUTE_COMPLETION_BONUS <= tail:
    raise ValueError("Wolf通关补偿不足以覆盖正向折现尾值")
  return tail


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
  t = value.clamp(0.0, 1.0)
  return t.square() * (3.0 - 2.0 * t)


def _low_speed_alpha(magnitude: torch.Tensor) -> torch.Tensor:
  # 低速段(<=0.4)采用50%细核，避免0.2m/s指令下静止不动也能拿到85%+满分。
  low_gate = 1.0 - _smoothstep((magnitude - 0.4) / 0.2)
  return 0.5 * low_gate


def track_linear_velocity_x(env, command_name, sigma=0.25, fine_sigma=0.02):
  command = env.command_manager.get_command(command_name)
  actual = env.scene["robot"].data.root_link_lin_vel_b
  error = (command[:, 0] - actual[:, 0]).square()
  alpha = _low_speed_alpha(command[:, :2].norm(dim=-1))
  return (1.0 - alpha) * torch.exp(-error / sigma) + alpha * torch.exp(
    -error / fine_sigma
  )


def track_linear_velocity_y(env, command_name, sigma=0.25, fine_sigma=0.02):
  command = env.command_manager.get_command(command_name)
  actual = env.scene["robot"].data.root_link_lin_vel_b
  error = (command[:, 1] - actual[:, 1]).square()
  alpha = _low_speed_alpha(command[:, :2].norm(dim=-1))
  return (1.0 - alpha) * torch.exp(-error / sigma) + alpha * torch.exp(
    -error / fine_sigma
  )


def track_angular_velocity_z(env, command_name, sigma=0.25, fine_sigma=0.02):
  command = env.command_manager.get_command(command_name)
  actual = env.scene["robot"].data.root_link_ang_vel_b
  error = (command[:, 2] - actual[:, 2]).square()
  alpha = _low_speed_alpha(command[:, 2].abs())
  return (1.0 - alpha) * torch.exp(-error / sigma) + alpha * torch.exp(
    -error / fine_sigma
  )


def privileged_terrain(env):
  """187个全地形高度及有效位只送给critic；actor仍只看相机可见几何。"""
  sensor = env.scene["terrain_scan"].data
  height = env.scene["robot"].data.root_link_pos_w[:, 2:3] - sensor.hit_pos_w[..., 2]
  valid = sensor.distances >= 0
  if (
    not torch.isfinite(sensor.distances).all()
    or not torch.isfinite(height[valid]).all()
  ):
    raise FloatingPointError("Wolf特权地形扫描非有限")
  return torch.cat(
    (torch.where(valid, height, 0.0).clamp(-2, 2), valid.to(height.dtype)), -1
  )


def support_height(env):
  """桥桩上按真实承重轮接触面定高度，不能被空隙底板扫描拉低。"""
  contact = env.scene["wheel_slip_contact"].data
  loaded = (contact.found > 0) & (contact.force[..., 0].abs() > 1.0)
  count = loaded.sum(-1)
  surface = torch.where(loaded, contact.pos[..., 2], 0.0).sum(-1) / count.clamp_min(1)
  clearance = env.scene["robot"].data.root_link_pos_w[:, 2] - surface
  return torch.where(count > 0, (clearance - 0.40).square(), 0.0)


def rolling_cost(env, sensor_name, asset_cfg, lateral=False):
  side, forward, _ = wheel_slip_velocities(env, sensor_name, asset_cfg)
  if lateral:
    # 原地转向与主动横移命令都需要横向擦滑；直行时保持全额侧滑约束。
    cmd = env.command_manager.get_command("twist")
    demand = cmd[:, 2].abs() + 2.0 * cmd[:, 1].abs()
    return side.square().sum(-1) / (1.0 + 4.0 * demand)
  return forward.square().sum(-1)


def stop_body(env):
  cmd = env.command_manager.get_command("twist")
  stopped = cmd.abs().amax(-1) < 1e-5
  robot = env.scene["robot"].data
  return stopped * (
    robot.root_link_lin_vel_b.square().sum(-1)
    + 0.5 * robot.root_link_ang_vel_b.square().sum(-1)
  )


def stop_wheels(env, asset_cfg):
  stopped = env.command_manager.get_command("twist").abs().amax(-1) < 1e-5
  speed = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  return stopped * speed.square().sum(-1)


def stop_legs(env, asset_cfg):
  stopped = env.command_manager.get_command("twist").abs().amax(-1) < 1e-5
  speed = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  return stopped * speed.square().sum(-1)


class RouteCompletionBonus:
  """一次性吸收态通关补偿，超过2.75*0.02/(1-0.99)=5.5的正向折现尾值上界。"""

  def __init__(self, cfg, env):
    del cfg
    self.paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.paid[ids] = False

  def __call__(self, env):
    command = env.command_manager.get_term("twist")
    command.evidence()
    eligible = (~command.flat_mask) & command.passed & (~command.failed) & (~self.paid)
    self.paid |= eligible
    return eligible.float() * (ROUTE_COMPLETION_BONUS / env.step_dt)


class RouteProgressShaping:
  """单调历史最远进度与出口1..49步稳停塑形，不可来回刷分。"""

  def __init__(self, cfg, env):
    del cfg
    self.best_progress = torch.zeros(env.num_envs, device=env.device)
    self.best_settled = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.best_progress[ids] = 0.0
    self.best_settled[ids] = 0
    self.initialized[ids] = False

  def __call__(self, env):
    from wolf.curriculum import SETTLE_CONTROLS

    command = env.command_manager.get_term("twist")
    command.evidence()
    rough = ~command.flat_mask
    valid = rough & (~command.failed) & torch.isfinite(command.last_root_progress)
    progress = torch.where(valid, command.last_root_progress, self.best_progress)
    fresh = valid & (~self.initialized)
    self.best_progress = torch.where(fresh, progress, self.best_progress)
    self.initialized |= fresh
    advance = (progress - self.best_progress).clamp(min=0.0)
    credible = valid & (~fresh) & (advance <= STALL_MAX_CREDIBLE_STEP_M)
    supported = command.last_contact_count >= 2
    progress_reward = (
      torch.where(credible & supported, advance, 0.0)
      * ROUTE_PROGRESS_PER_M
      / env.step_dt
    )
    self.best_progress = torch.where(
      valid, torch.maximum(self.best_progress, progress), self.best_progress
    )
    max_partial = max(1, SETTLE_CONTROLS - 1)
    settled = command.settled.clamp(min=0, max=max_partial)
    delta_settled = (settled - self.best_settled).clamp(min=0)
    delta_settled = torch.where(valid, delta_settled, 0)
    self.best_settled = torch.where(
      valid, self.best_settled + delta_settled, self.best_settled
    )
    settle_reward = (
      delta_settled.float() * (ROUTE_SETTLE_SHAPING_TOTAL / max_partial) / env.step_dt
    )
    return progress_reward + settle_reward


class RouteStallPenalty:
  """专项移动阶段长时间无有效前进时，按比例扣回正向速度跟踪奖励，防原地白嫖。"""

  def __init__(self, cfg, env):
    if cfg.weight != -1.0:
      raise ValueError("RouteStallPenalty权重必须为-1.0以仅抵消正向跟踪奖励")
    n, device = env.num_envs, env.device
    self.initialized = torch.zeros(n, dtype=torch.bool, device=device)
    self.age = torch.zeros(n, device=device)
    self.raw_best = torch.zeros(n, device=device)
    self.previous = torch.zeros(n, device=device)
    self.credit = torch.zeros(n, device=device)
    self.demand = torch.zeros(n, device=device)
    self.credit_history = torch.zeros(STALL_WINDOW_STEPS, n, device=device)
    self.demand_history = torch.zeros(STALL_WINDOW_STEPS, n, device=device)
    self.cursor = 0
    self.indices: list[int] | None = None

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.initialized[ids] = False
    self.age[ids] = 0.0
    self.raw_best[ids] = 0.0
    self.previous[ids] = 0.0
    self.credit[ids] = 0.0
    self.demand[ids] = 0.0
    self.credit_history[:, ids] = 0.0
    self.demand_history[:, ids] = 0.0

  def __call__(self, env):
    command = env.command_manager.get_term("twist")
    command.evidence()
    dt = env.step_dt
    rough = ~command.flat_mask
    progress = command.last_root_progress
    finite = torch.isfinite(progress)
    pos = torch.where(finite, progress, self.previous)
    fresh = rough & (~self.initialized)
    self.previous = torch.where(fresh, pos, self.previous)
    self.raw_best = torch.where(fresh, pos, self.raw_best)
    self.initialized |= fresh
    advance = (pos - self.raw_best).clamp(min=0.0)
    credible = (
      rough
      & finite
      & (~command.failed)
      & (command.last_contact_count >= 2)
      & ((pos - self.previous).abs() <= STALL_MAX_CREDIBLE_STEP_M)
    )
    self.credit += torch.where(credible, advance, 0.0)
    self.raw_best = torch.maximum(self.raw_best, pos)
    self.previous.copy_(pos)
    cmd_speed = command.command[:, :2].norm(dim=-1)
    moving = rough & finite & (cmd_speed > 0.05) & (env.episode_length_buf >= 50)
    self.demand += torch.where(moving, cmd_speed * dt, 0.0)
    window_credit = self.credit - self.credit_history[self.cursor]
    window_demand = self.demand - self.demand_history[self.cursor]
    self.credit_history[self.cursor].copy_(self.credit)
    self.demand_history[self.cursor].copy_(self.demand)
    self.cursor = (self.cursor + 1) % STALL_WINDOW_STEPS
    self.age += rough.float() * dt
    grace = self.age <= STALL_GRACE_SECONDS + 1e-6
    fraction = (
      (window_credit - STALL_DEADBAND_M).clamp(min=0.0)
      / (STALL_DEMAND_FRACTION * window_demand).clamp(min=1e-8)
    ).clamp(0.0, 1.0)
    gate = torch.where(moving & (~grace), fraction, 1.0)
    gate = torch.where(rough & (~finite), 0.0, gate)
    manager = env.reward_manager
    if self.indices is None:
      names = manager.active_terms
      self.indices = [names.index(k) for k in WOLF_POSITIVE_TRACKING_TERMS]
      if max(self.indices) >= names.index("route_stall_penalty"):
        raise ValueError("正向跟踪奖励必须在route_stall_penalty之前计算")
    positive = manager._step_reward[:, self.indices].sum(-1)
    return torch.where(rough, (1.0 - gate) * positive, 0.0)
