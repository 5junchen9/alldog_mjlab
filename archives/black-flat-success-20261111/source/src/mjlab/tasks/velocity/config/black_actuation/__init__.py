"""Opt-in Black ablations; original Stock registrations are untouched."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import black_actuation_flat_env_cfg, black_actuation_rough_env_cfg

for terrain, factory in (
  ("Flat", black_actuation_flat_env_cfg),
  ("Rough", black_actuation_rough_env_cfg),
):
  for suffix, friction, delay in (
    ("Friction", True, False),
    ("Delay", False, True),
    ("Actuation", True, True),
  ):
    rl_cfg = unitree_go1_ppo_runner_cfg()
    rl_cfg.experiment_name = f"black_stock_{suffix.lower()}_velocity_{terrain.lower()}"
    register_mjlab_task(
      task_id=f"Mjlab-Velocity-{terrain}-Black-Stock-{suffix}",
      env_cfg=factory(friction=friction, control_delay=delay),
      play_env_cfg=factory(play=True, friction=friction, control_delay=delay),
      rl_cfg=rl_cfg,
      runner_cls=VelocityOnPolicyRunner,
    )
