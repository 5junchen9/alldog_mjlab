"""固定3seed、确定性动作、逐控制步原始证据；默认打印计划。"""

import argparse
import fcntl
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from black.eval_collect import device_lock_path, write_json

from wolf.checkpoint import CONTRACT, load_bundle, source_identity, teacher_identity
from wolf.curriculum import COURSES
from wolf.eval_adapter import EvaluationAdapter
from wolf.eval_protocol import PROTOCOL, SEEDS, panel
from wolf.eval_score import digest, geometry_contract, score
from wolf.runtime import NativeEnv


def collect(env, teacher, specs, paths, max_controls=None):
  """每条lane只记录首个episode，首次reset后不混入新episode。"""
  active = [True] * len(specs)
  limit = max(row["duration"] for row in specs)
  if max_controls is not None:
    limit = min(limit, max_controls)
  streams = [path.open("x") for path in paths]
  try:
    with torch.inference_mode():
      for tick in range(limit):
        history = env.obs["history"].reshape(-1, 6, 53)
        commands = env.plant.command_manager.get_command("twist").cpu().clone()
        history_commands = history[:, -1, :3].cpu().clone()
        action = teacher(history, env.obs["geometry"])
        requested = action.cpu().clone()
        _, _, done, _ = env.step(action)
        post = {
          key: value.cpu().clone() for key, value in env.evaluation.closed.items()
        }
        for lane, (spec, stream) in enumerate(zip(specs, streams, strict=True)):
          if not active[lane]:
            continue
          ended = bool(done[lane]) or tick + 1 == spec["duration"]
          row = {
            key: post[key][lane].tolist()
            for key in (
              "velocity",
              "angular",
              "tilt",
              "contacts",
              "contact_points",
              "root",
              "processed_targets",
              "native_failure",
              "timeout",
              "lower_floor",
            )
          }
          row.update(
            tick=tick,
            command=commands[lane].tolist(),
            history_command=history_commands[lane].tolist(),
            action=requested[lane].tolist(),
            received_action=post["action"][lane].tolist(),
            raw_action=post["raw_action"][lane].tolist(),
            end=ended,
          )
          stream.write(json.dumps(row, allow_nan=False) + "\n")
          if ended:
            active[lane] = False
        if not any(active):
          break
  finally:
    for stream in streams:
      stream.close()


def execute(args):
  args.out.mkdir(parents=True, exist_ok=False)
  teacher, _, _, packet = load_bundle(args.bundle, args.device)
  teacher.freeze()
  identity = teacher_identity(teacher)
  sources = source_identity()
  specs = [row for seed in SEEDS for row in panel(args.course, seed, args.max_level)]
  specs = [{**row, "raw_file": f"trial_{i:04d}.jsonl"} for i, row in enumerate(specs)]
  manifest = {
    "contract": CONTRACT,
    "protocol": PROTOCOL,
    "teacher_id": identity,
    "source_identity": sources,
    "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
    "completed_updates": packet["completed_updates"],
    "course": args.course,
    "max_level": args.max_level,
    "max_controls": args.max_controls,
    "trials": specs,
    "geometry": {row["trial_id"]: geometry_contract(row) for row in specs},
  }
  write_json(args.out / "MANIFEST.json", manifest)
  env = None
  try:
    for seed in SEEDS:
      selected = [row for row in specs if row["seed"] == seed]
      for start in range(0, len(selected), 32):
        rows = selected[start : start + 32]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = NativeEnv(
          len(rows),
          seed,
          args.device,
          args.out,
          args.course,
          EvaluationAdapter(rows),
          max_level=args.max_level,
          allow_promotion=False,
        )
        env.policy_identity = identity
        collect(
          env,
          teacher,
          rows,
          [args.out / row["raw_file"] for row in rows],
          args.max_controls,
        )
        env.close()
        env = None
    if teacher_identity(teacher) != identity or source_identity() != sources:
      raise RuntimeError("采集期间教师或源码改变")
    result = {
      "manifest_sha256": digest(manifest),
      "teacher_id": identity,
      "optimizer_calls": 0,
      "teacher_unchanged": True,
      "raw_sha256": {
        row["trial_id"]: hashlib.sha256(
          (args.out / row["raw_file"]).read_bytes()
        ).hexdigest()
        for row in specs
      },
    }
    write_json(args.out / "RESULT.json", result)
    report = score(args.out)
    write_json(args.out / "SCORE.json", report)
    return report
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
  parser.add_argument("--course", choices=COURSES, default="flat_foundation")
  parser.add_argument(
    "--max-level", type=int, help="专项连续前缀最高关卡；general为1至4"
  )
  parser.add_argument("--bundle", type=Path)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--max-controls", type=int)
  parser.add_argument("--execute", action="store_true")
  args = parser.parse_args()
  from wolf.run import validate_cap

  validate_cap(parser, args.course, args.max_level)
  if args.max_controls is not None and args.max_controls < 1:
    parser.error("max-controls必须为正")
  if not args.execute:
    print(
      json.dumps(
        {
          "executed": False,
          "protocol": PROTOCOL,
          "panels": [panel(args.course, seed, args.max_level) for seed in SEEDS],
        },
        ensure_ascii=False,
        indent=2,
      )
    )
    return
  if args.bundle is None or args.out is None:
    parser.error("执行需要bundle和新的out目录")
  torch.set_num_threads(2)
  with device_lock_path(args.device).open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(json.dumps(execute(args), ensure_ascii=False))


if __name__ == "__main__":
  main()
