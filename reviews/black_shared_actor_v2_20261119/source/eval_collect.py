"""冻结教师的独立能力诊断；默认打印计划，显式执行才创建环境。"""

import argparse
import fcntl
import hashlib
import json
import os
import random
import tempfile
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import numpy as np
import torch
from checkpoint import CONTRACT, load_bundle, teacher_identity
from eval_adapter import EvaluationAdapter
from eval_protocol import (
  EVALUATION_SEEDS,
  HEIGHTS_CM,
  PROTOCOL,
  PROTOCOL_ID,
  QUALIFICATION_ID,
  SCHEMA,
  digest,
  padded_batch,
  roster,
)
from runtime import BASE, HERE, ROOT, NativeEnv


def sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
  """序列化和写盘都完成后才发布正式名称，已有文件拒覆盖。"""
  path = Path(path)
  data = json.dumps(
    value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
  )
  descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
  temporary = Path(name)
  try:
    with os.fdopen(descriptor, "w") as stream:
      stream.write(data + "\n")
      stream.flush()
      os.fsync(stream.fileno())
    os.link(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)
  directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
  try:
    os.fsync(directory)
  finally:
    os.close(directory)


def runtime_identity():
  """运行库版本不含安装目录、主机名或GPU序号，也不初始化CUDA。"""
  return {
    "python": python_version(),
    "packages": {
      name: version(name)
      for name in (
        "torch",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "rsl-rl-lib",
        "numpy",
        "tensordict",
      )
    },
    "torch_cuda_build": torch.version.cuda,
  }


def source_files():
  """相对路径绑定代码/资产；@runtime/versions.json是版本对象的虚拟键。"""
  directories = [
    HERE,
    BASE / "black_reliability_20261119/student",
    BASE / "black_reliability_20261119/fault",
    BASE / "stairs_passage_course_candidate_20261117",
    BASE / "terrain446_success12335_20261112",
    BASE / "comprehensive_lambda4_from8335_cap5000_20261112",
    BASE / "student_direct160_adapter_candidate_20261116",
    BASE
    / "structural_stoprepair_v7_fix1_candidate_20261115/candidate_source/preparation/rough_v2/specialty446_nativecontexts_v34_20261114",
  ]
  suffixes = {
    ".py",
    ".xml",
    ".urdf",
    ".stl",
    ".obj",
    ".msh",
    ".ply",
    ".mtl",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tga",
    ".webp",
    ".hdr",
  }
  files = {
    path
    for path in (ROOT / "src").rglob("*")
    if path.is_file() and path.suffix.lower() in suffixes
  }
  for directory in directories:
    files.update(directory.glob("*.py"))
  identities = {
    path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(files)
  }
  identities["@runtime/versions.json"] = digest(runtime_identity())
  return identities


def new_trial(spec, manifest) -> dict:
  keys = ("schema", "protocol_id", "teacher_id", "bundle_sha256", "source_sha256")
  trial: dict = {key: manifest[key] for key in keys}
  trial.update({key: spec[key] for key in ("trial_id", "batch_id", "seed", "lane")})
  trial.update(episode=0, birth=0, rows=[], end_reason="not_started")
  return trial


def collect_batch(env, teacher, specs, trials, max_controls=None):
  """每条lane只登记初次episode，首次终止后永不补入reset后的帧。"""
  limit = specs[0]["horizon"]
  if max_controls is not None:
    limit = min(limit, max_controls)
  active = [True] * len(specs)
  initial = env.evaluation.pre()
  for spec, trial in zip(specs, trials, strict=True):
    lane = spec["lane"]
    trial["episode"] = int(initial["episode"][lane])
    trial["birth"] = int(initial["tick"])
    trial["end_reason"] = "administrative_incomplete"
  with torch.inference_mode():
    for tick in range(limit):
      pre = env.evaluation.pre()
      if any(
        not bool(torch.isfinite(pre[key]).all())
        for key in ("velocity", "tilt", "command")
      ):
        raise FloatingPointError("评估动作前测量非有限")
      metadata = env.frames.metadata()
      # metadata含缓存tensor；必须在step的reset/新帧写入前保存。
      frame = metadata["frame"].cpu().clone()
      capture_tick = metadata["capture_tick"].cpu().clone()
      valid = metadata["mask"].cpu().clone()
      action = teacher(env.obs["history"].reshape(-1, 5, 45), env.obs["geometry"])
      if not bool(torch.isfinite(action).all()):
        raise FloatingPointError("评估教师动作非有限")
      actions = action.cpu().clone()
      env.step(action)
      post = env.evaluation.post()
      if any(
        not bool(torch.isfinite(post[key]).all())
        for key in ("action", "delayed_action", "processed_target")
      ):
        raise FloatingPointError("评估执行器证据非有限")
      if not torch.equal(post["action"], actions) or not torch.equal(
        post["raw_action"], actions
      ):
        raise RuntimeError("教师动作与原生执行器实际接收动作不同")
      post_episode = env.frames.episode.cpu()
      for index, (spec, trial) in enumerate(zip(specs, trials, strict=True)):
        if not active[index]:
          continue
        lane = spec["lane"]
        if int(pre["episode"][lane]) != trial["episode"]:
          raise RuntimeError("原trial已结束，不能接入新episode")
        terminated = bool(post["terminated"][lane])
        truncated = bool(post["truncated"][lane])
        failed = bool(post["failed"][lane])
        row: dict = dict(
          tick=tick,
          control_tick=int(pre["tick"]),
          episode=int(pre["episode"][lane]),
          post_episode=int(post_episode[lane]),
          command=pre["command"][lane].tolist(),
          velocity=pre["velocity"][lane].tolist(),
          tilt_rad=float(pre["tilt"][lane]),
          contacts=pre["foot_contacts"][lane].tolist(),
          support_proxy=bool(pre["support"][lane]),
          frame=int(frame[lane]),
          capture_tick=int(capture_tick[lane]),
          depth_valid=bool(valid[lane]),
          action=actions[lane].tolist(),
          delayed_action=post["delayed_action"][lane].tolist(),
          processed_target=post["processed_target"][lane].tolist(),
          action_delay=int(post["action_delay"][lane]),
          terminated=terminated,
          truncated=truncated,
          failed=failed,
        )
        strict = False
        if spec["kind"] == "stairs":
          strict = bool(post["strict_finished"][lane])
          row["stair"] = dict(
            ordered_treads=int(post["ordered_treads"][lane]),
            landing_contact=bool(post["landing"][lane]),
            pass_once=bool(post["pass_once"][lane]),
            invalid=bool(post["pass_invalid"][lane]),
            first_verified_pass_tick=int(post["first_verified_pass_tick"][lane]),
            birth=int(post["birth"][lane]),
            closed_tick=int(post["closed_tick"]),
            height_cm=int(post["height_cm"][lane]),
            direction=int(post["direction"][lane]),
            angle_deg=int(post["angle_deg"][lane]),
            safe_exit=bool(post["safe_exit"][lane]),
            strict_finish=strict,
            settled_steps=int(post["settled_steps"][lane]),
            linear_speed=float(post["post_linear_speed"][lane]),
            angular_speed=float(post["post_angular_speed"][lane]),
            tilt_rad=float(post["post_tilt"][lane]),
            contacts=post["post_contacts"][lane].tolist(),
          )
        trial["rows"].append(row)
        if terminated or truncated:
          if failed or (terminated and not strict):
            reason = "physical_failure"
          elif strict:
            reason = "strict_finish"
          else:
            reason = "native_timeout"
          trial["end_reason"] = reason
          active[index] = False
      if not any(active):
        break
  for index, (spec, trial) in enumerate(zip(specs, trials, strict=True)):
    if active[index] and len(trial["rows"]) == spec["horizon"]:
      trial["end_reason"] = "horizon"


def close_environment(env, error=None):
  """原始故障、故障证据写入和清理错误分别保存。"""
  if env is None:
    return
  try:
    env.close()
  except BaseException as cleanup:
    if error is None:
      raise
    error.__dict__.setdefault("failure_details", {})["close_error"] = repr(cleanup)


def execute(args, specs):
  output = args.out
  output.mkdir(parents=True, exist_ok=False)
  trials = []
  env = None
  failure = None
  try:
    teacher, _, _, packet = load_bundle(args.bundle, args.device)
    teacher.freeze()
    identity = teacher_identity(teacher)
    sources = source_files()
    manifest = dict(
      schema=SCHEMA,
      protocol_id=PROTOCOL_ID,
      protocol=PROTOCOL,
      teacher_id=identity,
      bundle_sha256=sha256(args.bundle),
      source_files=sources,
      source_sha256=digest(sources),
      runtime_identity=runtime_identity(),
      contract=CONTRACT,
      bundle=str(args.bundle.resolve()),
      completed_updates=packet["completed_updates"],
      roster=specs,
      action_mode="mean",
      qualification=QUALIFICATION_ID,
      max_controls=args.max_controls,
      suite=args.suite,
      base_seed=args.seed,
      batch_size=32,
      padding_scored=False,
    )
    write_json(output / "MANIFEST.json", manifest)
    trials = [new_trial(spec, manifest) for spec in specs]
    batches = []
    for start in range(0, len(specs), 32):
      batch = specs[start : start + 32]
      seed = batch[0]["seed"]
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      adapter = EvaluationAdapter(padded_batch(batch))
      batch_output = output / f"batch_{start // 32:03d}"
      batch_output.mkdir()
      env = NativeEnv(32, seed, args.device, batch_output, evaluation=adapter)
      env.policy_identity = identity
      batches.append(
        dict(
          batch_id=start // 32,
          camera=env.camera_receipt,
          render=env.render_receipt,
          trial_count=len(batch),
          padding_count=32 - len(batch),
        )
      )
      collect_batch(env, teacher, batch, trials[start : start + 32], args.max_controls)
      closing = env
      env = None
      close_environment(closing)
    if teacher_identity(teacher) != identity:
      raise RuntimeError("评估改变了教师权重")
    if source_files() != sources:
      raise RuntimeError("评估期间源码发生变化，不能发布完整结果")
    entries = []
    for trial in trials:
      path = output / f"{trial['trial_id']}.json"
      write_json(path, trial)
      entries.append(
        dict(trial_id=trial["trial_id"], path=path.name, sha256=sha256(path))
      )
    result = dict(
      schema=SCHEMA,
      manifest_sha256=sha256(output / "MANIFEST.json"),
      teacher_id=identity,
      bundle_sha256=manifest["bundle_sha256"],
      source_sha256=manifest["source_sha256"],
      action_mode="mean",
      optimizer_calls=0,
      teacher_unchanged=True,
      batches=batches,
      trials=entries,
      collection_complete=all(
        t["end_reason"] != "administrative_incomplete" for t in trials
      ),
      qualification=QUALIFICATION_ID,
    )
    write_json(output / "RESULT.json", result)
    return result
  except BaseException as error:
    failure = error
    details = error.__dict__.setdefault("failure_details", {})
    details["error"] = repr(error)
    if env is not None and env.probe is not None and env.probe.fault is not None:
      try:
        from fault_capture import write_fault

        write_fault(env.probe, output / "first_fault")
      except BaseException as diagnostic:
        details["diagnostic_error"] = repr(diagnostic)
    raise
  finally:
    close_environment(env, failure)
    if failure is not None:
      details = failure.__dict__["failure_details"]
      try:
        write_json(
          output / "PARTIAL.json",
          {"roster": specs, "trials": trials, "failure": details},
        )
      except BaseException as recording:
        details["partial_write_error"] = repr(recording)
      try:
        write_json(output / "FAILED.json", details)
      except BaseException as recording:
        details["failure_write_error"] = repr(recording)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", choices=("flat8", "axes", "stairs"), default="flat8")
  parser.add_argument("--seed", type=int, default=119041)
  parser.add_argument(
    "--panel",
    action="store_true",
    help="运行该suite的三个固定资格seed；每个seed独立目录",
  )
  parser.add_argument("--heights-cm", type=int, nargs="+", default=list(HEIGHTS_CM))
  parser.add_argument("--angles-deg", type=int, nargs="+", default=[-15, 0, 15])
  parser.add_argument("--bundle", type=Path)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--max-controls", type=int)
  parser.add_argument("--execute", action="store_true")
  args = parser.parse_args()
  if args.max_controls is not None and args.max_controls < 1:
    parser.error("--max-controls必须为正；截短结果保持incomplete")
  seeds = EVALUATION_SEEDS if args.panel else (args.seed,)
  plans = [roster(args.suite, seed, args.heights_cm, args.angles_deg) for seed in seeds]
  if not args.execute:
    print(json.dumps(dict(executed=False, protocol=PROTOCOL, panels=plans), indent=2))
    return
  if args.bundle is None or args.out is None:
    parser.error("执行需要--bundle及新的--out目录")
  torch.set_num_threads(2)
  from copy import copy

  from eval_score import score_attempt

  if args.panel:
    args.out.mkdir(parents=True, exist_ok=False)
  with Path("/tmp/mjlab_black_matrix_1000_cuda_0.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for seed, specs in zip(seeds, plans, strict=True):
      selected = copy(args)
      selected.seed = seed
      if args.panel:
        selected.out = args.out / f"seed_{seed}"
      result = execute(selected, specs)
      scored = score_attempt(
        selected.out / "MANIFEST.json", selected.out / "RESULT.json"
      )
      write_json(selected.out / "SCORE.json", scored)
      print(
        json.dumps(dict(trials=len(result["trials"]), score=scored), ensure_ascii=False)
      )


if __name__ == "__main__":
  main()
