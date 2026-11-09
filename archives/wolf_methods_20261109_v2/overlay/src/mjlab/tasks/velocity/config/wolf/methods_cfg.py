"""Opt-in Wolf methods configuration; existing four baseline tasks are unchanged.

Factories are the ablation boundary: switches independently remove each method.
Defaults are deliberately smaller than the 4096-env baseline for an 8 GB GPU;
this is a starting budget, NOT a verified hardware memory guarantee.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import cast

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity.config.wolf.env_cfg import wolf_rough_him_env_cfg
from mjlab.tasks.velocity.config.wolf.mdp.commands import WolfVelocityCommandCfg
from mjlab.tasks.velocity.config.wolf.mdp.curriculums import CURRICULUM_TERM_NAME
from mjlab.tasks.velocity.config.wolf.mdp.methods_observations import (
  configure_terrain_teacher_observations,
)
from mjlab.tasks.velocity.config.wolf.methods_terrain_cfg import (
  configure_terrain_teacher,
)
from mjlab.tasks.velocity.config.wolf.rl_cfg import (
  WolfHimAlgorithmCfg,
  WolfHimModelCfg,
  WolfHimRunnerCfg,
  wolf_him_runner_cfg,
)


@dataclass
class WolfMethodsModelCfg(WolfHimModelCfg):
  class_name: str = "mjlab.rl.wolf_him.perception_models:PerceptionHIMPolicy"
  use_him_estimator: bool = True
  distillation_enabled: bool = True
  prediction_enabled: bool = True
  methods_latent_dim: int = 16
  teacher_local_group: str = "terrain_teacher_local"
  teacher_global_group: str = "terrain_teacher_global"


@dataclass
class WolfMethodsAlgorithmCfg(WolfHimAlgorithmCfg):
  class_name: str = "mjlab.rl.wolf_him.methods_ppo:MethodsPPO"
  dual_critic_enabled: bool = True
  sparse_reward_key: str = "wolf_sparse_reward"
  advantage_weights: tuple[float, float] = (1.0, 1.0)
  auxiliary_learning_rate: float = 3e-4
  auxiliary_batch_size: int = 256
  auxiliary_max_samples: int = 2048
  distillation_loss_weight: float = 1.0
  prediction_loss_weight: float = 1.0


@dataclass
class WolfMethodsRunnerCfg(WolfHimRunnerCfg):
  actor: RslRlModelCfg = field(default_factory=WolfMethodsModelCfg)
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=WolfMethodsAlgorithmCfg)


def wolf_methods_env_cfg(
  play: bool = False,
  *,
  soft_terrain: bool = False,
  distillation: bool = True,
  foot_placement: bool = True,
  clf: bool = True,
):
  cfg = wolf_rough_him_env_cfg(play=play)
  cfg.scene.num_envs = 1 if play else 256
  # This SPECIALIST task assesses forward support placement, not turning.
  # Baseline flat/rough tasks retain their omnidirectional command distribution.
  command = cast(WolfVelocityCommandCfg, cfg.commands["twist"])
  command.ranges.lin_vel_x = (0.2, 0.8)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  cfg.curriculum.pop(CURRICULUM_TERM_NAME, None)
  configure_terrain_teacher(
    cfg, soft_terrain=soft_terrain, foot_placement=foot_placement, clf=clf
  )
  if distillation:
    configure_terrain_teacher_observations(cfg)
  return cfg


def wolf_methods_runner_cfg(
  *,
  use_him_estimator: bool = True,
  soft_terrain: bool = False,
  distillation: bool = True,
  prediction: bool = True,
  dual_critic: bool = True,
) -> WolfMethodsRunnerCfg:
  base = wolf_him_runner_cfg(stage="rough")
  kwargs = {item.name: getattr(base, item.name) for item in fields(base)}
  actor = asdict(base.actor)
  actor.update(
    class_name=WolfMethodsModelCfg.class_name,
    use_him_estimator=use_him_estimator,
    distillation_enabled=distillation,
    prediction_enabled=prediction,
  )
  algorithm = asdict(base.algorithm)
  algorithm.update(
    class_name=WolfMethodsAlgorithmCfg.class_name, dual_critic_enabled=dual_critic
  )
  # The symmetric actor still only consumes the original body history. Retain
  # privileged scans in the TensorDict, but never feed them to the actor loss.
  if algorithm["symmetry_cfg"] is not None:
    algorithm["symmetry_cfg"]["data_augmentation_func"] = (
      "mjlab.tasks.velocity.config.wolf.mdp.methods_observations:methods_left_right_mirror"
    )
  critic = asdict(base.critic)
  if dual_critic:
    critic["class_name"] = "mjlab.rl.wolf_him.dual_critic:DualCriticModel"
  kwargs.update(
    actor=WolfMethodsModelCfg(**actor),
    algorithm=WolfMethodsAlgorithmCfg(**algorithm),
    critic=RslRlModelCfg(**critic),
    run_name="wolf_methods_soft" if soft_terrain else "wolf_methods_hard",
    load_run=".*_wolf_methods_soft$" if soft_terrain else ".*_wolf_methods_hard$",
    num_steps_per_env=32,
    warm_start_supported=False,
    logger="tensorboard",
    upload_model=False,
  )
  return WolfMethodsRunnerCfg(**kwargs)
