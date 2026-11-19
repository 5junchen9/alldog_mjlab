"""显式的新版本入口。默认只显示计划，不创建环境或优化器。"""

import argparse
import dataclasses
import fcntl
import inspect
import json
import random
from pathlib import Path

import numpy as np
import torch
from checkpoint import (
  CONTRACT,
  initial_models,
  load_bundle,
  require_teacher_evaluation,
  save_bundle,
  teacher_identity,
  tensor_digest,
)
from learner_state import (
  SCHEMA,
  capture_rng,
  parameter_signature,
  restore_rng,
  validate_resume,
)
from ppo import build_ppo
from runtime import NativeEnv
from student import SharedStudent
from student_training import train_student


def cpu_copy(value):
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().clone()
  if isinstance(value, dict):
    return {key: cpu_copy(item) for key, item in value.items()}
  return value


def collect_probe(env, teacher, student, controller, controls):
  """只让指定控制者动作；教师标签在同一决策状态上计算。"""
  rows, hidden = [], None
  velocity_error = torch.zeros(3, device=env.device)
  velocity_count = 0
  resets = 0
  with torch.no_grad():
    for _ in range(controls):
      obs = env.obs
      history = obs["history"].reshape(-1, 5, 45)
      metadata = env.frames.metadata()
      teacher_action = teacher(history, obs["geometry"])
      terrain_feature = teacher.terrain_encoder(obs["geometry"])
      if controller == "student":
        action, _, hidden = student(history, env.frames.depth, metadata, hidden)
      else:
        action = teacher_action
      # 同一输入下共享head应给出相同动作；不以不同轨迹比较来代替。
      shared_action = student.act_with_visual(history, terrain_feature)
      torch.testing.assert_close(shared_action, teacher_action, rtol=0, atol=0)
      row = dict(
        obs=cpu_copy(dict(obs)),
        depth=cpu_copy(env.frames.depth),
        metadata=cpu_copy(metadata),
        teacher_action=cpu_copy(teacher_action),
        teacher_feature=cpu_copy(terrain_feature),
        requested_action=cpu_copy(action),
        controller=controller,
      )
      prediction = teacher.him(history)[:, :3]
      next_obs, reward, done, _ = env.step(action)
      # auto_reset后的终止行不能冒充上一episode的真实下一步速度。
      keep = ~done.bool()
      velocity_error += (prediction[keep] - next_obs["critic"][keep, :3]).abs().sum(0)
      velocity_count += int(keep.sum())
      resets += int(done.sum())
      row.update(reward=cpu_copy(reward), done=cpu_copy(done))
      rows.append(row)
  return rows, {
    "controls": controls,
    "environments": env.count,
    "optimizer_calls": 0,
    "controller": controller,
    "reset_events": resets,
    "shared_head_same_input_exact": True,
    "him_next_velocity_nonterminal_rows": velocity_count,
    "him_next_velocity_mae_xyz": (
      (velocity_error / velocity_count).cpu().tolist() if velocity_count else None
    ),
  }


def train_teacher(
  env, teacher, cnn, gru, rounds, output, *, resume=None, save_every=100
):
  """恢复仅限新版本学习器；物理、课程和相机从新episode开始。"""
  from rsl_rl.algorithms import PPO

  if rounds < 1 or save_every < 1:
    raise ValueError("新增轮数与存档间隔必须为正")
  accepted = set(inspect.signature(PPO.__init__).parameters)
  options = {
    key: value
    for key, value in dataclasses.asdict(env.rl.algorithm).items()
    if key in accepted
  }
  recipe = {
    "algorithm": "shared_actor_teacher_ppo_v1",
    "num_envs": env.count,
    "controls_per_round": 24,
    "ppo_options": options,
  }
  saved = validate_resume(resume, "teacher", recipe) if resume is not None else None
  algorithm = build_ppo(
    env.obs, teacher, num_steps=24, device=env.device, ppo_options=options
  )
  initial_him = tensor_digest(teacher.him.state_dict())
  optimizer_calls = 0
  completed = 0
  parameters = [
    (f"{prefix}.{name}", parameter)
    for prefix, module in (("actor", algorithm.actor), ("critic", algorithm.critic))
    for name, parameter in module.named_parameters()
    if parameter.requires_grad
  ]
  signature = parameter_signature(parameters)
  if saved is not None and resume is not None:
    if teacher_identity(teacher) != resume["teacher_id"]:
      raise ValueError("恢复必须使用同一bundle载入的teacher")
    if signature != saved["parameter_signature"]:
      raise ValueError("教师optimizer参数顺序不一致")
    algorithm.load(saved["ppo"], load_cfg=None, strict=True)
    # PPO包内也含teacher权重；不能让它悄悄覆盖外层已校验的模型身份。
    if teacher_identity(teacher) != resume["teacher_id"]:
      raise ValueError("PPO内的teacher权重与bundle身份不一致")
    completed = resume["completed_updates"]
    optimizer_calls = saved["optimizer_calls"]
    restore_rng(saved["rng"], env.device)

  def snapshot(path, update):
    return save_bundle(
      path,
      teacher,
      cnn,
      gru,
      phase="teacher",
      updates=update,
      learning_state={
        "schema": SCHEMA,
        "recipe": recipe,
        "ppo": algorithm.save(),
        "rng": capture_rng(env.device),
        "parameter_signature": signature,
        "optimizer_calls": optimizer_calls,
        "restart_boundary": "new_physics_curriculum_camera_episode",
      },
    )

  def counted(*_):
    nonlocal optimizer_calls
    optimizer_calls += 1

  def finite_gradients(optimizer, *_):
    gradients = [p.grad for group in optimizer.param_groups for p in group["params"]]
    if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
      raise FloatingPointError("PPO梯度缺失或非有限，停止更新")

  algorithm.optimizer.register_step_pre_hook(finite_gradients)
  algorithm.optimizer.register_step_post_hook(counted)
  for update in range(completed + 1, completed + rounds + 1):
    env.metrics.policy_update = update - 1
    velocity_error = torch.zeros(3, device=env.device)
    velocity_count = 0
    with torch.no_grad():
      for _ in range(24):
        history = env.obs["history"].reshape(-1, 5, 45)
        velocity_prediction = teacher.him(history)[:, :3]
        action = algorithm.act(env.obs)
        obs, reward, done, info = env.step(action)
        keep = ~done.bool()
        velocity_error += (
          (velocity_prediction[keep] - obs["critic"][keep, :3]).abs().sum(0)
        )
        velocity_count += int(keep.sum())
        algorithm.process_env_step(obs, reward, done, info)
      algorithm.compute_returns(env.obs)
    losses = algorithm.update()
    if tensor_digest(teacher.him.state_dict()) != initial_him:
      raise RuntimeError("教师训练改变了冻结HIM")
    with (output / "updates.jsonl").open("a") as stream:
      stream.write(
        json.dumps(
          {
            "update": update,
            "optimizer_calls": optimizer_calls,
            "him_next_velocity_rows": velocity_count,
            "him_next_velocity_mae_xyz": (velocity_error / velocity_count)
            .cpu()
            .tolist()
            if velocity_count
            else None,
            "losses": losses,
          }
        )
        + "\n"
      )
    # PPO已消费并清空该轮rollout，下一轮失败不会破坏此完整存档。
    if update % save_every == 0:
      snapshot(output / f"teacher_update_{update:06d}.pt", update)
  identity = snapshot(output / "teacher.pt", completed + rounds)
  return {
    "starting_updates": completed,
    "added_updates": rounds,
    "completed_updates": completed + rounds,
    "optimizer_calls": optimizer_calls,
    "teacher_id": identity,
    "qualification": "not_evaluated",
    "recipe": recipe,
    "learner_resumed": saved is not None,
    "exact_physical_resume": False,
    "restart_boundary": "new_physics_curriculum_camera_episode",
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--mode", choices=("plan", "probe", "teacher", "student"), default="plan"
  )
  parser.add_argument("--execute", action="store_true")
  parser.add_argument("--controller", choices=("teacher", "student"), default="teacher")
  parser.add_argument("--out", type=Path)
  source = parser.add_mutually_exclusive_group()
  source.add_argument("--bundle", type=Path, help="只加载模型权重，建立新学习器")
  source.add_argument(
    "--resume-learner", type=Path, help="恢复新版本学习器；物理冷启动"
  )
  parser.add_argument("--teacher-evaluation", type=Path)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--controls", type=int, default=24)
  parser.add_argument("--rounds", type=int)
  parser.add_argument("--save-every", type=int, default=100)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=119032)
  args = parser.parse_args()
  if args.mode == "plan":
    print(
      json.dumps(
        {"contract": CONTRACT, "executed": False, "formal_training": "stopped"},
        ensure_ascii=False,
        indent=2,
      )
    )
    return
  if not args.execute or args.out is None:
    parser.error("执行需要显式--execute和新的--out目录")
  if args.num_envs < 16 or args.controls < 1:
    parser.error("至少16个环境，控制步数必须为正")
  if args.mode == "probe" and args.controls > 120:
    parser.error("接线探针最多120控制步")
  if args.mode in ("teacher", "student") and (args.rounds is None or args.rounds < 1):
    parser.error("更新次数必须通过--rounds显式指定")
  if args.save_every < 1:
    parser.error("--save-every必须为正，单位为完整round")
  if args.resume_learner and args.mode not in ("teacher", "student"):
    parser.error("--resume-learner只用于teacher或student更新")
  if args.mode == "student" and (
    (args.bundle is None and args.resume_learner is None)
    or args.teacher_evaluation is None
  ):
    parser.error("学生训练需要模型来源和该教师的--teacher-evaluation")
  args.out.mkdir(parents=True, exist_ok=False)
  env = None
  failure = None
  with Path("/tmp/mjlab_black_matrix_1000_cuda_0.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
      random.seed(args.seed)
      np.random.seed(args.seed)
      torch.manual_seed(args.seed)
      torch.set_num_threads(2)
      selected = args.resume_learner or args.bundle
      resume = None
      if selected:
        teacher, cnn, gru, packet = load_bundle(selected, args.device)
        if args.resume_learner:
          resume = packet
      else:
        teacher, cnn, gru = initial_models(args.device)
      evaluation = None
      if args.mode == "student":
        # 在创建仿真环境和学生优化器前核验所选教师，不能绕过验收直接蒸馏。
        evaluation = require_teacher_evaluation(
          args.teacher_evaluation, teacher_identity(teacher)
        )
      before = tensor_digest(teacher.state_dict())
      env = NativeEnv(args.num_envs, args.seed, args.device, args.out)
      if args.mode == "probe":
        teacher.freeze()
        student = SharedStudent(teacher, cnn, gru).eval()
        rows, result = collect_probe(
          env, teacher, student, args.controller, args.controls
        )
        torch.save(rows, args.out / "trajectory.pt")
        result["teacher_weights_unchanged"] = before == tensor_digest(
          teacher.state_dict()
        )
        result["teacher_id"] = teacher_identity(teacher)
        save_bundle(
          args.out / "candidate.pt", teacher, cnn, gru, phase="probe", updates=0
        )
      elif args.mode == "teacher":
        result = train_teacher(
          env,
          teacher,
          cnn,
          gru,
          args.rounds,
          args.out,
          resume=resume,
          save_every=args.save_every,
        )
      else:
        result = train_student(
          env,
          teacher,
          cnn,
          gru,
          args.rounds,
          args.out,
          resume=resume,
          save_every=args.save_every,
        )
        result["teacher_evaluation"] = evaluation
      result.update(
        camera=env.camera_receipt,
        render=env.render_receipt,
        initialization=(
          "learner_resume_cold_environment"
          if args.resume_learner
          else "bundle_weights_only"
          if args.bundle
          else "r800_weights_only"
        ),
        exact_failure_replay=False,
        capability_evaluated=False,
      )
      (args.out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
      print(json.dumps(result))
    except BaseException as error:
      failure = error.__dict__.get("failure_details", {})
      failure["error"] = repr(error)
      # 即使磁盘写入也失败，调用方仍能从原异常读取次级诊断。
      error.__dict__["failure_details"] = failure
      if env is not None and env.probe is not None and env.probe.fault is not None:
        try:
          from probe import write_fault

          write_fault(env.probe, args.out / "first_fault")
        except BaseException as diagnostic_error:
          failure["diagnostic_error"] = repr(diagnostic_error)
      raise
    finally:
      try:
        try:
          if env is not None:
            env.close()
        except BaseException as close_error:
          if failure is None:
            failure = {"error": repr(close_error)}
            close_error.__dict__["failure_details"] = failure
            raise
          failure["close_error"] = repr(close_error)
      finally:
        if failure is not None:
          try:
            (args.out / "FAILED.json").write_text(json.dumps(failure) + "\n")
          except BaseException as record_error:
            failure["failure_write_error"] = repr(record_error)


if __name__ == "__main__":
  main()
