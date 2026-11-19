"""Wolf教师PPO/HIM交替更新；视觉学生复用统一的50帧重叠记忆训练。"""

import json
from pathlib import Path

import torch
from black.checkpoint import tensor_digest
from black.learner_state import (
  SCHEMA,
  capture_rng,
  parameter_signature,
  restore_rng,
)
from black.learner_state import (
  validate_resume as validate_shared_resume,
)
from black.run import update_him

from wolf.checkpoint import save_bundle, source_identity, teacher_identity
from wolf.ppo import build_ppo

STEPS = 24


def validate_resume(packet, phase, configuration):
  """完整学习器续训绑定原运行库，跨机只允许载入权重再重新验收。"""
  if packet["source_identity"] != source_identity():
    raise ValueError("完整Wolf学习器恢复需要相同源码、资产及运行库")
  return validate_shared_resume(
    packet, phase, configuration, current_source=source_identity()
  )


def recipe(env, phase):
  return {
    "algorithm": "wolf_shared_actor_v3",
    "phase": phase,
    "course": env.course,
    "max_level": env.plant.command_manager.get_term("twist").cfg.max_level,
    "num_envs": env.count,
    "controls_per_round": STEPS,
    "HIM_target": "next_velocity_scaled2_and_next_frame_before_reset",
    "HIM_microbatch": "balanced_max2048",
    "HIM_temperature": 0.5,
    "ppo": "lr1e-3_adaptive_KL0.01_gamma0.99",
    "camera_noise": getattr(env, "training_sensors", False),
  }


def train_teacher(env, teacher, cnn, gru, rounds, output, resume=None, save_every=100):
  output = Path(output)
  teacher.him.freeze()
  algorithm = build_ppo(env.obs, teacher, device=env.device)
  optimizer = torch.optim.Adam(teacher.him.parameters(), lr=1e-4)
  configuration = recipe(env, "teacher")
  signature = parameter_signature(
    [
      (f"{prefix}.{name}", p)
      for prefix, module in (("actor", algorithm.actor), ("critic", algorithm.critic))
      for name, p in module.named_parameters()
      if p.requires_grad
    ]
  )
  completed = calls = him_calls = 0
  if resume is not None:
    state = validate_resume(resume, "teacher", configuration)
    if state["parameter_signature"] != signature:
      raise ValueError("Wolf教师optimizer参数顺序变化")
    algorithm.load(state["ppo"], load_cfg=None, strict=True)
    if teacher_identity(teacher) != resume["teacher_id"]:
      raise ValueError("Wolf PPO内teacher权重与bundle身份不一致")
    optimizer.load_state_dict(state["him_optimizer"])
    completed, calls, him_calls = (
      resume["completed_updates"],
      state["optimizer_calls"],
      state["him_optimizer_calls"],
    )
    env.restore_curriculum(state["curriculum"])
    restore_rng(state["rng"], env.device)

  def count(*_):
    nonlocal calls
    calls += 1

  def finite(opt, *_):
    if any(
      p.grad is None or not torch.isfinite(p.grad).all()
      for group in opt.param_groups
      for p in group["params"]
    ):
      raise FloatingPointError("Wolf PPO梯度缺失/非有限")

  algorithm.optimizer.register_step_pre_hook(finite)
  algorithm.optimizer.register_step_post_hook(count)

  def save(path):
    return save_bundle(
      path,
      teacher,
      cnn,
      gru,
      phase="teacher",
      updates=completed,
      learning_state={
        "schema": SCHEMA,
        "recipe": configuration,
        "ppo": algorithm.save(),
        "him_optimizer": optimizer.state_dict(),
        "parameter_signature": signature,
        "optimizer_calls": calls,
        "him_optimizer_calls": him_calls,
        "rng": capture_rng(env.device),
        "curriculum": env.capture_curriculum(),
      },
    )

  for _ in range(rounds):
    samples = []
    env.metrics.policy_update = completed
    env.policy_identity = teacher_identity(teacher)
    before = tensor_digest(teacher.him.state_dict())
    with torch.no_grad():
      for _ in range(STEPS):
        history = env.obs["history"].reshape(-1, 6, 53).clone()
        action = algorithm.act(env.obs)
        obs, reward, done, info = env.step(action)
        keep = ~info["terminated"].bool()
        if keep.any():
          values = history, info["terminal_history"][:, -1], info["next_velocity"]
          samples.append(tuple(value[keep].detach().cpu().clone() for value in values))
        algorithm.process_env_step(obs, reward, done, info)
      algorithm.compute_returns(env.obs)
    losses = algorithm.update()
    if before != tensor_digest(teacher.him.state_dict()):
      raise RuntimeError("PPO改变了冻结HIM")
    actor_before = tensor_digest(teacher.actor.state_dict())
    terrain_before = tensor_digest(teacher.terrain_encoder.state_dict())
    auxiliary = update_him(teacher.him, optimizer, samples, env.device)
    him_calls += auxiliary["optimizer_calls"]
    if actor_before != tensor_digest(
      teacher.actor.state_dict()
    ) or terrain_before != tensor_digest(teacher.terrain_encoder.state_dict()):
      raise RuntimeError("HIM辅助更新改变了actor/地形encoder")
    completed += 1
    with (output / "updates.jsonl").open("a") as stream:
      stream.write(
        json.dumps(
          {
            "update": completed,
            "ppo": losses,
            "HIM": auxiliary,
            "optimizer_calls": calls,
            "him_optimizer_calls": him_calls,
          }
        )
        + "\n"
      )
    if completed % save_every == 0:
      save(output / f"teacher_update_{completed:06d}.pt")
  identity = save(output / "teacher.pt")
  return {
    "teacher_id": identity,
    "completed_updates": completed,
    "optimizer_calls": calls,
    "him_optimizer_calls": him_calls,
    "checkpoint": str(output / "teacher.pt"),
  }


def train_student(env, teacher, cnn, gru, rounds, output, resume=None, save_every=100):
  from black.student_training import train_student as train_visual

  from wolf import checkpoint

  return train_visual(
    env,
    teacher,
    cnn,
    gru,
    rounds,
    output,
    resume=resume,
    save_every=save_every,
    checkpoint_api=checkpoint,
  )
