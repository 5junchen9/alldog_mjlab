"""两阶段教师—学生训练的第一阶段：平地特权教师。"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.black_actuation.env_cfgs import (
  black_actuation_flat_env_cfg,
)

from .omni_command import OmniVelocityCommandCfg


def black_teacher_flat_env_cfg(
  play: bool = False,
  *,
  friction: bool = False,
  control_delay: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = black_actuation_flat_env_cfg(
    play=play,
    friction=friction,
    control_delay=control_delay,
  )
  # 原critic已包含72维真值：本体状态、线速度和足端状态。
  # 教师actor读取同样的信息，两份配置独立，后续修改互不影响。
  cfg.observations["actor"] = deepcopy(cfg.observations["critic"])
  # 真值教师关闭观测噪声；保留原环境中的摩擦、质心、推力等随机化。
  cfg.observations["actor"].enable_corruption = False
  return cfg


def black_teacher_omni_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = black_teacher_flat_env_cfg(play=play, friction=True, control_delay=True)
  cfg.commands["twist"] = OmniVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    heading_command=False,
    rel_heading_envs=0.0,
    rel_forward_envs=0.0,
    rel_standing_envs=0.0,
    debug_vis=True,
    ranges=OmniVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-0.5, 0.5),
    ),
  )
  cfg.curriculum.pop("command_vel", None)
  return cfg
