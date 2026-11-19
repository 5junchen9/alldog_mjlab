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

from black.checkpoint import (
  CONTRACT,
  initial_models,
  load_bundle,
  require_teacher_evaluation,
  save_bundle,
  source_identity,
  teacher_identity,
  tensor_digest,
)
from black.learner_state import (
  SCHEMA,
  capture_rng,
  parameter_signature,
  restore_rng,
  validate_resume,
)
from black.model import make_student_modules
from black.ppo import build_ppo
from black.runtime import NativeEnv
from black.student import SharedStudent
from black.student_training import train_student


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


def update_him(him, optimizer, samples, device):
  """PPO全部结束后才改变HIM；真实终止行不作为下一帧监督。"""
  count = sum(len(row[0]) for row in samples)
  if not count:
    return {"rows": 0, "optimizer_calls": 0}
  history, next_frame, next_velocity = [
    torch.cat([row[column] for row in samples]) for column in range(3)
  ]
  optimizer.zero_grad(set_to_none=True)
  him.enable_learning()
  velocity_sum = swap_sum = 0.0
  try:
    for start in range(0, count, 2048):
      stop = min(start + 2048, count)
      velocity_loss, swap_loss = him.estimator_losses(
        history[start:stop].to(device),
        next_frame[start:stop].to(device),
        next_velocity[start:stop].to(device),
      )
      weight = (stop - start) / count
      loss = (velocity_loss + swap_loss) * weight
      if not bool(torch.isfinite(loss)):
        raise FloatingPointError("HIM损失非有限，停止更新")
      loss.backward()
      velocity_sum += float(velocity_loss.detach()) * weight
      swap_sum += float(swap_loss.detach()) * weight
    parameters = list(him.parameters())
    if any(p.grad is None for p in parameters):
      raise FloatingPointError("HIM梯度缺失，停止更新")
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
  finally:
    him.freeze()
  return {
    "rows": count,
    "optimizer_calls": 1,
    "velocity_huber": velocity_sum,
    "swap_loss": swap_sum,
  }


def train_teacher(
  env, teacher, cnn, gru, rounds, output, *, resume=None, save_every=100
):
  """完整一轮=PPO更新+HIM辅助更新；恢复时物理和课程重开episode。"""
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
    "algorithm": "black_retrain_teacher_ppo_v1",
    "course": env.course,
    "task": getattr(env, "task", None),
    "num_envs": env.count,
    "controls_per_round": 24,
    "ppo_options": options,
    "critic_seed": teacher.initialization["critic_seed"],
    "him": {
      "update": "one_Adam_after_entire_PPO_round",
      "target": "reset_before_next_velocity_and_next_frame_exclude_terminated",
      "learning_rate": 1e-4,
      "loss": "Huber(next_velocity)+prototype_swap",
      "microbatch": 2048,
      "gradient_clip": 1.0,
    },
  }
  saved = (
    validate_resume(resume, "teacher", recipe, current_source=source_identity())
    if resume is not None
    else None
  )
  teacher.him.freeze()
  algorithm = build_ppo(
    env.obs, teacher, num_steps=24, device=env.device, ppo_options=options
  )
  him_parameters = list(teacher.him.named_parameters())
  him_signature = parameter_signature(him_parameters)
  him_optimizer = torch.optim.Adam([p for _, p in him_parameters], lr=1e-4)
  optimizer_calls = him_optimizer_calls = completed = 0
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
    if him_signature != saved["him_parameter_signature"]:
      raise ValueError("HIM optimizer参数顺序不一致")
    algorithm.load(saved["ppo"], load_cfg=None, strict=True)
    if teacher_identity(teacher) != resume["teacher_id"]:
      raise ValueError("PPO内的teacher权重与bundle身份不一致")
    him_optimizer.load_state_dict(saved["him_optimizer"])
    completed = resume["completed_updates"]
    optimizer_calls = saved["optimizer_calls"]
    him_optimizer_calls = saved["him_optimizer_calls"]
    if (
      type(him_optimizer_calls) is not int or not 0 <= him_optimizer_calls <= completed
    ):
      raise ValueError("HIM optimizer计数不一致")
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
        "him_optimizer": him_optimizer.state_dict(),
        "him_parameter_signature": him_signature,
        "him_optimizer_calls": him_optimizer_calls,
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
    env.policy_identity = teacher_identity(teacher)
    initial_him = tensor_digest(teacher.him.state_dict())
    samples = []
    velocity_error = torch.zeros(3, device=env.device)
    velocity_count = 0
    with torch.no_grad():
      for _ in range(24):
        history = env.obs["history"].reshape(-1, 5, 45).clone()
        prediction = teacher.him(history)[:, :3]
        action = algorithm.act(env.obs)
        obs, reward, done, info = env.step(action)
        # 纯超时是真实有效末帧；真实失败的终止行不进入辅助监督。
        keep = ~info["terminated"].bool()
        next_velocity = info["next_velocity"]
        next_frame = info["terminal_history"].reshape(-1, 5, 45)[:, -1]
        if bool(keep.any()):
          samples.append(
            tuple(
              cpu_copy(value[keep]) for value in (history, next_frame, next_velocity)
            )
          )
        velocity_error += (prediction[keep] - next_velocity[keep]).abs().sum(0)
        velocity_count += int(keep.sum())
        algorithm.process_env_step(obs, reward, done, info)
      algorithm.compute_returns(env.obs)
    losses = algorithm.update()
    if tensor_digest(teacher.him.state_dict()) != initial_him:
      raise RuntimeError("采样或PPO阶段改变了冻结HIM")
    actor_before_him = tensor_digest(teacher.actor.state_dict())
    terrain_before_him = tensor_digest(teacher.terrain_encoder.state_dict())
    him_metrics = update_him(teacher.him, him_optimizer, samples, env.device)
    him_optimizer_calls += him_metrics["optimizer_calls"]
    if actor_before_him != tensor_digest(
      teacher.actor.state_dict()
    ) or terrain_before_him != tensor_digest(teacher.terrain_encoder.state_dict()):
      raise RuntimeError("HIM辅助更新改变了actor或地形编码器")
    with (output / "updates.jsonl").open("a") as stream:
      stream.write(
        json.dumps(
          {
            "update": update,
            "optimizer_calls": optimizer_calls,
            "him_optimizer_calls": him_optimizer_calls,
            "him_frozen_during_rollout_and_PPO": True,
            "actor_unchanged_during_HIM_update": True,
            "him_changed_after_auxiliary": initial_him
            != tensor_digest(teacher.him.state_dict()),
            "him_next_velocity_rows": velocity_count,
            "him_next_velocity_mae_xyz": (velocity_error / velocity_count)
            .cpu()
            .tolist()
            if velocity_count
            else None,
            "him_losses": him_metrics,
            "losses": losses,
          }
        )
        + "\n"
      )
    if update % save_every == 0:
      snapshot(output / f"teacher_update_{update:06d}.pt", update)
  identity = snapshot(output / "teacher.pt", completed + rounds)
  return {
    "starting_updates": completed,
    "added_updates": rounds,
    "completed_updates": completed + rounds,
    "optimizer_calls": optimizer_calls,
    "him_optimizer_calls": him_optimizer_calls,
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
  parser.add_argument(
    "--course",
    choices=("flat_foundation", "flat_axes", "general", "specialty"),
    default="flat_foundation",
  )
  from support.black.config import TASKS

  parser.add_argument("--task", choices=TASKS)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--controls", type=int, default=24)
  parser.add_argument("--rounds", type=int)
  parser.add_argument("--save-every", type=int, default=100)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=119032)
  args = parser.parse_args()
  if (args.course == "specialty") != (args.task is not None):
    parser.error("specialty必须指定--task，其他阶段不接收专项task")
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
  if args.num_envs < 28 or args.num_envs % 4 or args.controls < 1:
    parser.error("至少28个环境且为4的倍数，控制步数必须为正")
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
  from black.curriculum import RESERVED_SEEDS

  if args.seed in RESERVED_SEEDS:
    parser.error("训练种子不能使用独立验收保留种子")
  if (
    args.mode == "teacher"
    and args.course != "flat_foundation"
    and not args.resume_learner
  ):
    if args.bundle is None or args.teacher_evaluation is None:
      parser.error("阶段切换需要--bundle及当前教师的--teacher-evaluation")
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
          args.teacher_evaluation,
          teacher_identity(teacher),
          "student:" + (args.task if args.course == "specialty" else args.course),
        )
        if not args.resume_learner:
          visual_seed = args.seed + 2
          with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(
              torch.Generator(device="cpu").manual_seed(visual_seed).get_state()
            )
            cnn, gru = make_student_modules()
          cnn, gru = cnn.to(args.device), gru.to(args.device)
          teacher.initialization["student_visual"] = {
            "mode": "scratch_at_student_start",
            "seed": visual_seed,
          }
      elif (
        args.mode == "teacher"
        and args.course != "flat_foundation"
        and not args.resume_learner
      ):
        evaluation = require_teacher_evaluation(
          args.teacher_evaluation,
          teacher_identity(teacher),
          "specialty:" + args.task if args.course == "specialty" else args.course,
        )
      before = tensor_digest(teacher.state_dict())
      max_level = None
      if args.mode == "student" and args.task is not None:
        assert evaluation is not None
        max_level = evaluation["max_levels"][args.task]
      env = NativeEnv(
        args.num_envs,
        args.seed,
        args.device,
        args.out,
        course=args.course,
        task=args.task,
        max_level=max_level,
      )
      env.policy_identity = teacher_identity(teacher)
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
        course=args.course,
        task=args.task,
        stage_admission=evaluation,
        initialization=(
          "learner_resume_cold_environment"
          if args.resume_learner
          else "bundle_teacher_fresh_student_visual"
          if args.mode == "student"
          else "bundle_weights_only"
          if args.bundle
          else "scratch_all_modules"
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
          from black.fault_capture import write_fault

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
