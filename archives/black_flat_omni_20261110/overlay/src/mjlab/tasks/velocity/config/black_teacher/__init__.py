"""独立注册平地教师任务，保留现有Black Stock任务。"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import black_teacher_flat_env_cfg, black_teacher_omni_env_cfg
from .rl_cfg import black_teacher_ppo_runner_cfg

for suffix, friction, delay in (
  ("", False, False),
  ("-Friction", True, False),
  ("-Delay", False, True),
  ("-Actuation", True, True),
):
  rl_cfg = black_teacher_ppo_runner_cfg()
  if suffix:
    rl_cfg.experiment_name += suffix.lower().replace("-", "_")
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-Flat-Black-Teacher{suffix}",
    env_cfg=black_teacher_flat_env_cfg(friction=friction, control_delay=delay),
    play_env_cfg=black_teacher_flat_env_cfg(
      play=True, friction=friction, control_delay=delay
    ),
    rl_cfg=rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
  )

# 单个教师、单个PPO：所有运动类型共同采样和更新。
omni_rl_cfg = black_teacher_ppo_runner_cfg()
omni_rl_cfg.experiment_name = "black_teacher_omni_flat"
omni_rl_cfg.max_iterations = 3000
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Black-Teacher-Omni",
  env_cfg=black_teacher_omni_env_cfg(),
  play_env_cfg=black_teacher_omni_env_cfg(play=True),
  rl_cfg=omni_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)
