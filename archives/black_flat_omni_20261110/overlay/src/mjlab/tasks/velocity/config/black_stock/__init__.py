"""Independent stock-baseline Black tasks; no old Black/Wolf imports."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import black_stock_flat_env_cfg, black_stock_rough_env_cfg

for terrain, factory in (
  ("Flat", black_stock_flat_env_cfg),
  ("Rough", black_stock_rough_env_cfg),
):
  rl_cfg = unitree_go1_ppo_runner_cfg()
  rl_cfg.experiment_name = f"black_stock_velocity_{terrain.lower()}"
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-{terrain}-Black-Stock",
    env_cfg=factory(),
    play_env_cfg=factory(play=True),
    rl_cfg=rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
  )
