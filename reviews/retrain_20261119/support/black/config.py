"""直接构造Black配置；不加载旧trainer、注册ID或权重。"""

import copy
from dataclasses import fields

from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.black_rough.env_cfgs import rough_env_cfg
from mjlab.tasks.velocity.config.black_teacher.rl_cfg import (
  black_teacher_ppo_runner_cfg,
)
from mjlab.utils.noise import UniformNoiseCfg
from support.black.observations import append_terrain
from support.black.specialty_composition import CompositionCommandCfg

TASKS = ("stairs", "platform", "high_wall", "slope", "broken_bridge", "plum_stakes")


def configure_ppo_schedule(rl, course="flat_foundation"):
  """按训练阶段配置PPO学习率与自适应KL调度；从零阶段不用旧微调5e-5。"""
  if course in ("flat_foundation", "flat_axes"):
    rl.algorithm.learning_rate = 1.0e-3
  elif course == "general":
    rl.algorithm.learning_rate = 3.0e-4
  elif course == "specialty":
    rl.algorithm.learning_rate = 1.0e-4
  else:
    raise ValueError(f"Unknown Black course for PPO schedule: {course}")
  rl.algorithm.schedule = "adaptive"
  rl.algorithm.desired_kl = 0.01
  return rl


def make_general_configs(
  num_envs=4096, seed=119032, map_seed=112001, profile_seed=None
):
  """25%平地与75%六种粗糙地形，沿用原生奖励和地形升级。"""
  cfg = rough_env_cfg(num_envs, seed, map_seed)
  # 新从零训练线在episode重置时清空执行器延迟历史，避免上一回合极端动作串扰新回合。
  cfg.actions["joint_pos"].clear_history_on_reset = True
  old = cfg.commands["twist"]
  cfg.commands["twist"] = CompositionCommandCfg(
    **{field.name: copy.deepcopy(getattr(old, field.name)) for field in fields(old)}
  )
  cfg.commands["twist"].profile_seed = seed if profile_seed is None else profile_seed
  cfg.sim.ccd_scratch_slots = 1024
  append_terrain(cfg)
  rl = black_teacher_ppo_runner_cfg()
  rl.seed = seed
  configure_ppo_schedule(rl, "general")
  rl.experiment_name = "black_general"
  return cfg, rl


comprehensive_configs = make_general_configs


def history_observations():
  """教师和学生共用同一带噪声的5帧本体历史。"""
  return ObservationGroupCfg(
    terms={
      "angular_velocity": ObservationTermCfg(
        func=mdp.base_ang_vel, scale=0.25, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2)
      ),
      "gravity": ObservationTermCfg(
        func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
      ),
      "joint_position": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"biased": True},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
      ),
      "joint_velocity": ObservationTermCfg(
        func=mdp.joint_vel_rel, scale=0.05, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5)
      ),
      "command": ObservationTermCfg(
        func=mdp.generated_commands, params={"command_name": "twist"}
      ),
      "previous_action": ObservationTermCfg(func=mdp.last_action),
    },
    concatenate_terms=True,
    enable_corruption=True,
    history_length=5,
    flatten_history_dim=False,
  )
