"""Black adaptation with locally editable copies of the stock Go1 rewards."""

import math

from mjlab.asset_zoo.robots.black_stock.black_constants import (
  BLACK_ACTION_SCALE,
  FOOT_NAMES,
  get_black_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.go1.env_cfgs import (
  unitree_go1_flat_env_cfg,
  unitree_go1_rough_env_cfg,
)


def _black_rewards(rough: bool) -> dict[str, RewardTermCfg]:
  """Black奖励的编辑入口：完整复制Go1配置，保留Black足端绑定。"""
  rewards = {
    # 跟踪平移速度指令。
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    # 跟踪转向角速度指令。
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    # 保持机身直立；Rough下参照地形法向。
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
      },
    ),
    # 按站立、行走、奔跑速度约束关节姿态。
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {
          r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
          r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
        },
        "std_walking": {
          r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
          r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
        },
        "std_running": {
          r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
          r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    # 以下零权重项沿用Go1设置，当前不参与奖励计算。
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk",))},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,
      params={"sensor_name": "robot/root_angmom"},
    ),
    # 惩罚关节越限和相邻控制步的动作变化。
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    # 足端腾空时间，Go1原配权重为零。
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
    # 足端离地高度与摆动过程的最大高度。
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_NAMES),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    # 惩罚接触时足端打滑和落地冲击。
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_NAMES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
  }

  # Go1仅在Rough任务中保留这三项碰撞惩罚。
  if rough:
    rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
    rewards["self_collisions"] = RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "self_collision"},
    )
    rewards["shank_collision"] = RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "shank_ground_touch"},
    )
    rewards["trunk_head_collision"] = RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "trunk_ground_touch"},
    )
  return rewards


def _adapt_black(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.scene.entities["robot"] = get_black_robot_cfg()
  # Initial zero-pose Black model requires 328 constraints before reset.
  # Flat Go1's 300 slots overflow; 400 is a small-run startup floor, not a
  # demonstrated capacity bound for arbitrary training/contact conditions.
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 400)
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = BLACK_ACTION_SCALE
  # Black has one thigh box, three shank cylinders and six trunk boxes.
  patterns = {
    "feet_ground_contact": tuple(f"{leg}_foot_collision" for leg in FOOT_NAMES),
    "thigh_ground_touch": tuple(f"{leg}_thigh_collision0" for leg in FOOT_NAMES),
    "shank_ground_touch": tuple(
      f"{leg}_calf_collision{i}" for leg in FOOT_NAMES for i in range(3)
    ),
    "trunk_ground_touch": tuple(f"trunk_collision{i}" for i in range(6)),
  }
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=leg, entity="robot") for leg in FOOT_NAMES
      )
    if sensor.name in patterns:
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = patterns[sensor.name]
  return cfg


def black_stock_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _adapt_black(unitree_go1_flat_env_cfg(play=play))
  cfg.rewards = _black_rewards(rough=False)
  return cfg


def black_stock_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _adapt_black(unitree_go1_rough_env_cfg(play=play))
  cfg.rewards = _black_rewards(rough=True)
  return cfg
