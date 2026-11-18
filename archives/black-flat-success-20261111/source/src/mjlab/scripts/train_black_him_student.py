"""加载已训练教师，在学生访问的状态上训练HIM+Huber学生。"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
import tyro
from tensordict import TensorDict
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.black_dagger import DAggerDataset
from mjlab.rl.black_him_student import HIMStudent, HIMStudentPolicy, update_student
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.velocity.config.black_teacher.student_env_cfg import (
  black_him_student_env_cfg,
)
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


@dataclass
class StudentTrainConfig:
  teacher_checkpoint: Path
  teacher_task: str = "Mjlab-Velocity-Flat-Black-Teacher-Actuation"
  num_envs: int = 1024
  iterations: int = 10000
  device: str = "cuda:0"
  seed: int = 42
  log_root: Path = Path("logs/black_him_student")
  resume: Path | None = None
  dagger_capacity: int = 100000


@torch.no_grad()
def collect_student_batch(
  env: RslRlVecEnvWrapper,
  teacher: nn.Module,
  student: HIMStudent,
  obs: TensorDict,
  *,
  use_teacher: bool = False,
) -> tuple[TensorDict, dict[str, torch.Tensor]]:
  samples: dict[str, list[torch.Tensor]] = {
    "history": [],
    "teacher_action": [],
    "next_frame": [],
    "next_velocity": [],
  }
  for _ in range(24):
    history = cast(torch.Tensor, obs["history"]).clone()
    # 教师为学生此刻到达的同一状态提供动作标签，未使用教师轨迹的时间索引。
    teacher_action = teacher(obs).clone()
    if env.clip_actions is not None:
      teacher_action.clamp_(-env.clip_actions, env.clip_actions)
    # 第一轮由教师带着采集，后续完全由学生控制。
    actions = teacher_action if use_teacher else student(history)
    next_obs, _, dones, _ = env.step(actions)
    samples["history"].append(history)
    samples["teacher_action"].append(teacher_action)
    # auto_reset=False：这些仍是真实终止帧，复制后才能reset。
    samples["next_frame"].append(cast(torch.Tensor, next_obs["history"])[:, -1].clone())
    samples["next_velocity"].append(
      cast(torch.Tensor, next_obs["actor"])[:, :3].clone()
    )
    ids = dones.nonzero().flatten()
    if len(ids):
      env.unwrapped.reset(env_ids=ids)
      # reset已更新观测缓存；此处读取缓存，不再次推进历史。
      obs = env.get_observations()
    else:
      obs = next_obs
  return obs, {name: torch.cat(values) for name, values in samples.items()}


def train_student(cfg: StudentTrainConfig) -> Path:
  torch.manual_seed(cfg.seed)
  env_cfg = black_him_student_env_cfg(cfg.teacher_task)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  teacher_cfg = load_rl_cfg(cfg.teacher_task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device),
    clip_actions=teacher_cfg.clip_actions,
  )
  log_dir = cfg.log_root / (
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + cfg.teacher_task
  )
  log_dir.mkdir(parents=True)
  writer = SummaryWriter(str(log_dir))
  try:
    teacher_runner = VelocityOnPolicyRunner(env, asdict(teacher_cfg), device=cfg.device)
    teacher_runner.load(str(cfg.teacher_checkpoint), map_location=cfg.device)
    teacher = teacher_runner.get_inference_policy().requires_grad_(False)
    student = HIMStudent().to(cfg.device)
    actor_optimizer = torch.optim.Adam(student.actor.parameters(), lr=1e-3)
    estimator_optimizer = torch.optim.Adam(student.estimator_parameters(), lr=1e-3)
    dagger = DAggerDataset(capacity=cfg.dagger_capacity)
    start_iteration = 0
    if cfg.resume is not None:
      # 聚合数据集保留在CPU，网络和优化器加载时再搬到各自设备。
      saved = torch.load(cfg.resume, map_location="cpu", weights_only=False)
      if (
        saved["teacher_checkpoint"] != str(cfg.teacher_checkpoint.resolve())
        or saved["teacher_task"] != cfg.teacher_task
      ):
        raise ValueError("恢复学生训练时需使用原教师checkpoint和任务。")
      student.load_state_dict(saved["student"])
      actor_optimizer.load_state_dict(saved["actor_optimizer"])
      estimator_optimizer.load_state_dict(saved["estimator_optimizer"])
      dagger.load_state_dict(saved["dagger_dataset"])
      start_iteration = saved["iteration"] + 1
      env.unwrapped.common_step_counter = saved["common_step_counter"]

    # 加载教师/学生的课程计数后，再生成新回合和完整历史。
    obs, _ = env.reset()
    metadata = {
      "config": asdict(cfg),
      "history_shape": [5, 45],
      "history_order": "oldest_to_newest",
      "proprio_order": [
        "angular_velocity*0.25",
        "projected_gravity",
        "joint_position_relative",
        "joint_velocity*0.05",
        "command",
        "previous_raw_action",
      ],
      "foot_order": ["FL", "FR", "RR", "RL"],
      "regression_loss": "Huber(delta=1.0)",
      "contrastive_loss": "HIM prototype swapped cross entropy",
      "dagger": {
        "capacity": dagger.capacity,
        "storage_device": "cpu",
        "aggregation": "uniform subset of all rounds via fixed random priorities",
        "teacher_rollout_first_round": True,
      },
    }
    (log_dir / "config.json").write_text(
      json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    for iteration in range(start_iteration, start_iteration + cfg.iterations):
      teacher_rollout = dagger.total_seen == 0
      obs, batch = collect_student_batch(
        env, teacher, student, obs, use_teacher=teacher_rollout
      )
      dagger.add(batch["history"], batch["teacher_action"])
      totals = {"action_huber": 0.0, "velocity_huber": 0.0, "him_swap": 0.0}
      for _ in range(5):
        indices = torch.randperm(batch["history"].shape[0], device=cfg.device)
        for ids in indices.chunk(4):
          imitation = dagger.sample(len(ids), cfg.device)
          losses = update_student(
            student,
            actor_optimizer,
            estimator_optimizer,
            history=batch["history"][ids],
            teacher_action=imitation["teacher_action"],
            next_frame=batch["next_frame"][ids],
            next_velocity=batch["next_velocity"][ids],
            dagger_history=imitation["history"],
          )
          for name, value in losses.items():
            totals[name] += value / 20
      for name, value in totals.items():
        writer.add_scalar(f"Loss/{name}", value, iteration)
      writer.add_scalar("DAgger/dataset_size", len(dagger), iteration)
      writer.add_scalar("DAgger/total_seen", dagger.total_seen, iteration)
      writer.add_scalar("DAgger/teacher_rollout", int(teacher_rollout), iteration)
      print(
        f"student iteration={iteration} losses={totals} "
        f"dagger_size={len(dagger)} total_seen={dagger.total_seen} "
        f"teacher_rollout={teacher_rollout}",
        flush=True,
      )
      if iteration % 50 == 0 or iteration == start_iteration + cfg.iterations - 1:
        torch.save(
          {
            "student": student.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "estimator_optimizer": estimator_optimizer.state_dict(),
            "dagger_dataset": dagger.state_dict(),
            "iteration": iteration,
            "common_step_counter": env.unwrapped.common_step_counter,
            "teacher_task": cfg.teacher_task,
            "teacher_checkpoint": str(cfg.teacher_checkpoint.resolve()),
          },
          log_dir / f"student_{iteration}.pt",
        )
        policy = HIMStudentPolicy(student).cpu().eval()
        torch.jit.script(policy).save(str(log_dir / "student_policy.pt"))
    return log_dir
  finally:
    writer.close()
    env.close()


if __name__ == "__main__":
  train_student(tyro.cli(StudentTrainConfig))
