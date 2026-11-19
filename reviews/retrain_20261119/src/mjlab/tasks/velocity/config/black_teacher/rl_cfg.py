"""平地教师使用普通PPO，沿用Go1网络和优化参数。"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg


def black_teacher_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.experiment_name = "black_teacher_velocity_flat"
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  return cfg
