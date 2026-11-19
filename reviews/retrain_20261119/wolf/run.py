"""Wolf从零共享actor训练；默认仅展示计划，显式execute才运行。"""

import argparse
import fcntl
import json
import random
from pathlib import Path

import numpy as np
import torch
from black.eval_collect import write_json
from black.student import SharedStudent

from wolf.checkpoint import CONTRACT, load_bundle, save_bundle, teacher_identity
from wolf.curriculum import COURSES, SPECIALTIES
from wolf.eval_score import require_qualification
from wolf.model import initial_models, make_student_modules
from wolf.runtime import NativeEnv
from wolf.training import train_student, train_teacher


def validate_cap(parser, course, cap):
  if course in SPECIALTIES:
    from support.black.obstacle_geometry import GROUPS

    if cap is None or not 0 <= cap < len(GROUPS[course]):
      parser.error(
        f"{course}需要显式max-level，范围0至{len(GROUPS[course]) - 1}；代表连续前缀"
      )
  elif course == "general":
    if cap is not None and not 1 <= cap <= 4:
      parser.error("general的max-level范围1至4")
  elif cap is not None:
    parser.error("平地课程不使用max-level")


def probe(env, teacher, cnn, gru, controls, controller):
  student = SharedStudent(teacher, cnn, gru).eval()
  identity = teacher_identity(teacher)
  hidden = None
  resets = 0
  with torch.inference_mode():
    for _ in range(controls):
      history = env.obs["history"].reshape(-1, 6, 53)
      action = teacher(history, env.obs["geometry"])
      shared = student.act_with_visual(
        history, teacher.terrain_encoder(env.obs["geometry"])
      )
      torch.testing.assert_close(action, shared, atol=0, rtol=0)
      if controller == "student":
        action, _, hidden = student(
          history, env.frames.depth, env.frames.metadata(), hidden
        )
      _, _, done, _ = env.step(action)
      resets += int(done.sum())
  if teacher_identity(teacher) != identity:
    raise RuntimeError("探针改变了教师")
  return {
    "controls": controls,
    "reset_events": resets,
    "optimizer_calls": 0,
    "shared_actor_exact": True,
    "teacher_unchanged": True,
  }


def execute(args):
  args.out.mkdir(parents=True, exist_ok=False)
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  selected = args.resume_learner or args.bundle
  resume = None
  if selected:
    teacher, cnn, gru, packet = load_bundle(selected, args.device)
    if args.resume_learner:
      resume = packet
  else:
    teacher, cnn, gru = initial_models(args.seed)
    teacher, cnn, gru = [module.to(args.device) for module in (teacher, cnn, gru)]
  qualification = None
  if args.mode == "student":
    qualification = require_qualification(
      args.qualification,
      teacher_identity(teacher),
      "student:" + args.course,
      args.max_level,
    )
    if resume is None:
      with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 2)
        cnn, gru = make_student_modules()
      cnn, gru = cnn.to(args.device), gru.to(args.device)
      teacher.initialization["student_visual"] = {
        "mode": "scratch_at_student_start",
        "seed": args.seed + 2,
      }
  elif args.mode == "teacher" and args.course != "flat_foundation" and resume is None:
    qualification = require_qualification(
      args.qualification, teacher_identity(teacher), args.course, args.max_level
    )
  env = None
  try:
    env = NativeEnv(
      args.num_envs,
      args.seed,
      args.device,
      args.out,
      args.course,
      max_level=args.max_level,
      allow_promotion=args.mode == "teacher",
    )
    env.policy_identity = teacher_identity(teacher)
    if args.mode == "probe":
      result = probe(env, teacher, cnn, gru, args.controls, args.controller)
      save_bundle(
        args.out / "candidate.pt", teacher, cnn, gru, phase="probe", updates=0
      )
    elif args.mode == "teacher":
      result = train_teacher(
        env, teacher, cnn, gru, args.rounds, args.out, resume, args.save_every
      )
    else:
      result = train_student(
        env, teacher, cnn, gru, args.rounds, args.out, resume, args.save_every
      )
    result.update(
      course=args.course,
      max_level=args.max_level,
      mode=args.mode,
      qualification=qualification,
      camera=env.camera_receipt,
      source="new_shared_actor_from_scratch"
      if selected is None
      else "same_contract_bundle",
      exact_physical_resume=False,
      capability_evaluated=False,
    )
    write_json(args.out / "RESULT.json", result)
    return result
  except BaseException as error:
    details = {"error": repr(error), **error.__dict__.get("failure_details", {})}
    if env is not None and env.probe is not None and env.probe.fault is not None:
      from wolf.fault_capture import write_fault

      write_fault(env.probe, args.out / "first_fault")
    write_json(args.out / "FAILED.json", details)
    raise
  finally:
    if env is not None:
      env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--mode", choices=("probe", "teacher", "student"), default="probe"
  )
  parser.add_argument("--course", choices=COURSES, default="flat_foundation")
  parser.add_argument("--max-level", type=int)
  parser.add_argument("--seed", type=int, default=1911700)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--controls", type=int, default=120)
  parser.add_argument("--controller", choices=("teacher", "student"), default="teacher")
  parser.add_argument("--rounds", type=int, default=1)
  parser.add_argument("--save-every", type=int, default=100)
  parser.add_argument("--qualification", type=Path)
  source = parser.add_mutually_exclusive_group()
  source.add_argument("--bundle", type=Path)
  source.add_argument("--resume-learner", type=Path)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--execute", action="store_true")
  args = parser.parse_args()
  validate_cap(parser, args.course, args.max_level)
  if min(args.num_envs, args.controls, args.rounds, args.save_every) < 1:
    parser.error("环境数、步数、轮数和存档间隔必须为正")
  if not args.execute:
    print(
      json.dumps(
        {
          "executed": False,
          "course_order": list(COURSES),
          "contract": CONTRACT,
          "selected": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
          },
        },
        ensure_ascii=False,
        indent=2,
      )
    )
    return
  if args.out is None:
    parser.error("执行需要新的out目录")
  if args.mode == "student" and not (args.bundle or args.resume_learner):
    parser.error("学生训练需要已独立评估的同合同教师bundle")
  if (
    args.mode == "teacher"
    and args.course != "flat_foundation"
    and not (args.bundle or args.resume_learner)
  ):
    parser.error("后续课程需要前一阶段同合同教师bundle")
  torch.set_num_threads(2)
  with Path("/tmp/mjlab_black_matrix_1000_cuda_0.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(json.dumps(execute(args), ensure_ascii=False))


if __name__ == "__main__":
  main()
