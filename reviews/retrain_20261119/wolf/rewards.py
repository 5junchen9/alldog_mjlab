"""轮胎材料接触点滑移与停车成本，正常纯滚动不受滑移惩罚。"""

import torch

from mjlab.tasks.velocity.config.wolf.mdp.common import wheel_slip_velocities


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
    # 轮足原地转向需要横向擦滑；转向时减弱侧滑成本，不能将其等同失稳。
    yaw = env.command_manager.get_command("twist")[:, 2].abs()
    return side.square().sum(-1) / (1.0 + 4.0 * yaw)
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
