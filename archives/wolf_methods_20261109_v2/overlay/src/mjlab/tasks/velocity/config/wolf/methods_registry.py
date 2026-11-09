"""Two separate specialist tasks; no original Wolf registration is replaced."""

from dataclasses import asdict

from mjlab.tasks.registry import register_mjlab_task

from .methods_cfg import wolf_methods_env_cfg, wolf_methods_runner_cfg
from .methods_runner import WolfMethodsOnPolicyRunner, validate_methods_profile

METHOD_TASKS = {"wolf-methods-soft": True, "wolf-methods-hard": False}


def make_methods_profile(
  *,
  soft_terrain=False,
  distillation=True,
  prediction=True,
  use_him_estimator=True,
  dual_critic=True,
  foot_placement=True,
  clf=True,
  play=False,
):
  """Build matched environment/network switches for a controlled ablation."""
  env = wolf_methods_env_cfg(
    play=play,
    soft_terrain=soft_terrain,
    distillation=distillation,
    foot_placement=foot_placement,
    clf=clf,
  )
  agent = wolf_methods_runner_cfg(
    soft_terrain=soft_terrain,
    distillation=distillation,
    prediction=prediction,
    use_him_estimator=use_him_estimator,
    dual_critic=dual_critic,
  )
  agent.experiment_name = "wolf_methods"
  validate_methods_profile(env, asdict(agent))
  return env, agent


for task_id, soft in METHOD_TASKS.items():
  env_cfg, rl_cfg = make_methods_profile(soft_terrain=soft)
  play_cfg, _ = make_methods_profile(soft_terrain=soft, play=True)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg,
    play_env_cfg=play_cfg,
    rl_cfg=rl_cfg,
    runner_cls=WolfMethodsOnPolicyRunner,
  )
