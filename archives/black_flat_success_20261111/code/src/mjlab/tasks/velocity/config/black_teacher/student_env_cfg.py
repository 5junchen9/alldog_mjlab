"""给教师环境增加学生可读取的5帧本体历史。"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg


def black_him_student_env_cfg(teacher_task: str) -> ManagerBasedRlEnvCfg:
  cfg = load_env_cfg(teacher_task)
  # 先读取真实终止帧作为HIM下一帧标签，再由学生采集循环手动reset。
  cfg.auto_reset = False
  cfg.observations["history"] = ObservationGroupCfg(
    terms={
      "angular_velocity": ObservationTermCfg(
        func=mdp.base_ang_vel,
        scale=0.25,
        noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
      ),
      "gravity": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
      ),
      "joint_position": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"biased": True},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
      ),
      "joint_velocity": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        scale=0.05,
        noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
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
  return cfg
