"""Prepare or explicitly run isolated Black Omni PPO seed experiments.

The default is a side-effect-free plan. Vector environments are rollout workers,
not independent policy seeds. This module uses only the standard library.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REWARD_NAMES = {
  "track_linear_velocity",
  "track_angular_velocity",
  "upright",
  "pose",
  "body_ang_vel",
  "angular_momentum",
  "dof_pos_limits",
  "action_rate_l2",
  "air_time",
  "foot_clearance",
  "foot_swing_height",
  "foot_slip",
  "soft_landing",
}
TASK = "Mjlab-Velocity-Flat-Black-Teacher-Omni"
SPEED_TASKS = {
  "Mjlab-Velocity-Flat-Black-Teacher-LowSpeed",
  "Mjlab-Velocity-Flat-Black-Teacher-HighSpeed",
}
CANDIDATE_TASKS = {
  f"Mjlab-Velocity-Flat-Black-Teacher-{speed}-{variant}"
  for speed in ("LowSpeed", "HighSpeed")
  for variant in ("Candidate", "CommandOnly")
}
ROOT = Path(__file__).resolve().parents[3]
RESUME_SCOPE = (
  "Restores policy, critic, optimizer, normalization and common_step_counter. "
  "New simulator episodes and RNG streams are created; not a full simulator/RNG "
  "snapshot or a bitwise continuation. No checkpoint is shared between seeds."
)
TEMPLATE = {
  "schema_version": 1,
  "experiment_id": "black_omni_choose_matrix",
  "iterations": 2000,
  "num_envs": 4096,
  "device": "cuda:0",
  "min_free_gpu_mb": 12000,
  "min_free_disk_gb": 10,
  "save_interval": 100,
  "selection_rule": "",
  "selection_seeds": [],
  "final_seeds": [],
  "eval_replicates": 16,
  "eval_pushes": False,
  "configs": [],
}


def read_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text())
  if not isinstance(value, dict):
    raise ValueError(f"Expected a JSON object: {path}")
  return value


def digest(value: Any) -> str:
  return hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()


def file_sha(path: Path) -> str:
  hasher = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      hasher.update(block)
  return hasher.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  with tmp.open("x") as stream:
    stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(tmp, path)


def integer(value: Any, name: str, low: int = 1) -> int:
  if type(value) is not int or value < low:
    raise ValueError(f"{name} must be an integer >= {low}")
  return value


def seeds(value: Any, name: str) -> list[int]:
  if not isinstance(value, list) or not value:
    raise ValueError(f"Fill in a nonempty {name} list; no seed count is assumed")
  for seed in value:
    integer(seed, name, 0)
    if seed >= 2**31:
      raise ValueError(f"{name} seeds must be below 2**31")
  if len(set(value)) != len(value):
    raise ValueError(f"Duplicate seeds in {name}")
  return value


def validate_config(cfg: dict[str, Any]) -> None:
  if set(cfg) != set(TEMPLATE):
    raise ValueError(f"Config fields must be exactly {sorted(TEMPLATE)}")
  if cfg["schema_version"] != 1:
    raise ValueError("Unsupported schema_version")
  if not isinstance(cfg["experiment_id"], str) or not re.fullmatch(
    r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", cfg["experiment_id"]
  ):
    raise ValueError("Invalid experiment_id")
  integer(cfg["iterations"], "iterations", 2000)
  for key in ("num_envs", "save_interval", "eval_replicates"):
    integer(cfg[key], key)
  integer(cfg["min_free_gpu_mb"], "min_free_gpu_mb", 1)
  integer(cfg["min_free_disk_gb"], "min_free_disk_gb", 1)
  if cfg["device"] != "cpu" and not re.fullmatch(r"cuda:\d+", cfg["device"]):
    raise ValueError("device must be cpu or one physical cuda:N GPU")
  if type(cfg["eval_pushes"]) is not bool:
    raise ValueError("eval_pushes must be a boolean")
  if not isinstance(cfg["selection_rule"], str) or not cfg["selection_rule"].strip():
    raise ValueError("Write the checkpoint selection rule before training")
  seen = set(seeds(cfg["selection_seeds"], "selection_seeds"))
  final = set(seeds(cfg["final_seeds"], "final_seeds"))
  if seen & final:
    raise ValueError("Selection and final evaluation seeds overlap")
  seen |= final
  rows = cfg["configs"]
  if not isinstance(rows, list) or not rows:
    raise ValueError("Fill in m configuration rows; no final m is assumed")
  ids, n = set(), None
  for row in rows:
    if (
      not isinstance(row, dict)
      or not {"id", "train_seeds", "overrides"} <= set(row)
      or set(row) - {"id", "train_seeds", "overrides", "task"}
    ):
      raise ValueError("Each row needs id, train_seeds and overrides")
    if not isinstance(row["id"], str) or not re.fullmatch(
      r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", row["id"]
    ):
      raise ValueError("Invalid config id")
    if row["id"] in ids:
      raise ValueError("Duplicate config id")
    ids.add(row["id"])
    train = seeds(row["train_seeds"], f"{row['id']}.train_seeds")
    if seen & set(train):
      raise ValueError("All train/selection/final seed sets must be disjoint")
    seen.update(train)
    n = len(train) if n is None else n
    if len(train) != n:
      raise ValueError("Every configuration must have the same n train seeds")
    task = row.get("task", TASK)
    if task not in {TASK, *SPEED_TASKS, *CANDIDATE_TASKS}:
      raise ValueError(
        "Only registered independent Black Omni/speed/candidate tasks are supported"
      )
    overrides = row["overrides"]
    if not isinstance(overrides, dict) or set(overrides) - {
      "learning_rate",
      "entropy_coef",
      "reward_weights",
      "speed",
    }:
      raise ValueError(
        "Allowed overrides: learning_rate, entropy_coef, reward_weights, speed"
      )
    if task in SPEED_TASKS:
      validate_speed(overrides.get("speed"))
    elif "speed" in overrides:
      raise ValueError(
        "Original Omni speed distribution is preserved; choose a speed task"
      )
    for key, value in overrides.items():
      if key == "speed":
        continue
      if key == "reward_weights":
        if not isinstance(value, dict):
          raise ValueError("reward_weights must be an object")
        if set(value) - REWARD_NAMES:
          raise ValueError("Unknown Black reward name")
        values = value.values()
      else:
        values = [value]
      for number in values:
        if type(number) not in (int, float) or not math.isfinite(number):
          raise ValueError("Overrides must contain finite numbers")
      if (
        key in ("learning_rate", "entropy_coef")
        and isinstance(value, (int, float))
        and value < 0
      ):
        raise ValueError(f"{key} must be nonnegative")
      if key == "learning_rate" and value == 0:
        raise ValueError("learning_rate must be positive")


def validate_speed(speed: Any) -> None:
  if not isinstance(speed, dict) or set(speed) != {"min_speeds", "max_speeds"}:
    raise ValueError("Speed projects require explicit min_speeds and max_speeds")
  for name in ("min_speeds", "max_speeds"):
    values = speed[name]
    if not isinstance(values, list) or len(values) != 3:
      raise ValueError(
        f"{name} must contain [vx, vy, yaw] limits, not a default preset"
      )
    for number in values:
      if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
        raise ValueError("Directional speed limits must be finite and positive")
  if any(
    low > high
    for low, high in zip(speed["min_speeds"], speed["max_speeds"], strict=True)
  ):
    raise ValueError("Minimum directional speed cannot exceed maximum")


def apply_overrides(env_cfg: Any, agent_cfg: Any, job: dict[str, Any]) -> None:
  """Apply only explicit matrix parameters to fresh registry configs."""
  overrides = job["overrides"]
  for key in ("learning_rate", "entropy_coef"):
    if key in overrides:
      setattr(agent_cfg.algorithm, key, overrides[key])
  for name, weight in overrides.get("reward_weights", {}).items():
    if name not in env_cfg.rewards:
      raise ValueError(f"Unknown Black reward {name}")
    env_cfg.rewards[name].weight = weight
  if job["task"] in SPEED_TASKS:
    speed = overrides["speed"]
    validate_speed(speed)
    command = env_cfg.commands["twist"]
    command.min_speeds = tuple(speed["min_speeds"])
    for axis, maximum in zip(
      ("lin_vel_x", "lin_vel_y", "ang_vel_z"), speed["max_speeds"], strict=True
    ):
      setattr(command.ranges, axis, (-maximum, maximum))


def source_files(root: Path = ROOT) -> list[Path]:
  files = [p for p in (root / "src").rglob("*") if p.is_file()]
  files = [p for p in files if "__pycache__" not in p.parts and p.suffix != ".pyc"]
  files += [root / name for name in ("pyproject.toml", "uv.lock", "run.sh")]
  return sorted(files)


def source_manifest(root: Path = ROOT) -> dict[str, str]:
  return {str(p.relative_to(root)): file_sha(p) for p in source_files(root)}


def make_plan(
  cfg: dict[str, Any], output: Path, sources: dict[str, str]
) -> dict[str, Any]:
  validate_config(cfg)
  jobs = []
  for row in cfg["configs"]:
    for seed in row["train_seeds"]:
      run_id = f"{row['id']}/seed_{seed}"
      jobs.append(
        {
          "run_id": run_id,
          "config_id": row["id"],
          "task": row.get("task", TASK),
          "training_seed": seed,
          "overrides": row["overrides"],
          "directory": str(output.resolve() / run_id),
        }
      )
  payload = {"config": cfg, "sources": sources, "jobs": jobs, "task": TASK}
  return {
    **payload,
    "fingerprint": digest(payload),
    "m": len(cfg["configs"]),
    "n": len(cfg["configs"][0]["train_seeds"]),
    "total_independent_trainings": len(jobs),
    "parallel_jobs": 1,
    "resume_scope": RESUME_SCOPE,
    "output": str(output.resolve()),
  }


@contextlib.contextmanager
def lock(path: Path):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a+") as handle:
    try:
      fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
      raise RuntimeError(f"Another process holds {path}") from error
    try:
      yield handle.fileno()
    finally:
      fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def guard_resources(cfg: dict[str, Any], output: Path) -> None:
  existing = output
  while not existing.exists():
    existing = existing.parent
  if shutil.disk_usage(existing).free < cfg["min_free_disk_gb"] * 1024**3:
    raise RuntimeError("Insufficient free disk; nothing launched")
  if cfg["device"] == "cpu":
    return
  gpu = cfg["device"].split(":")[1]
  common = ["nvidia-smi", "-i", gpu]
  active = subprocess.check_output(
    common + ["--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    text=True,
    timeout=10,
  ).strip()
  # The Linux desktop remote-display daemon appears as C+G in nvidia-smi.
  # It is not a training job; its memory still counts against the free limit.
  active = "\n".join(
    pid
    for pid in active.splitlines()
    if not (
      pid.strip().isdigit()
      and Path(f"/proc/{pid.strip()}/exe").resolve()
      == Path("/usr/libexec/gnome-remote-desktop-daemon")
    )
  )
  if active:
    raise RuntimeError(f"GPU {gpu} has compute processes ({active}); not launching")
  free = int(
    subprocess.check_output(
      common + ["--query-gpu=memory.free", "--format=csv,noheader,nounits"],
      text=True,
      timeout=10,
    ).strip()
  )
  if free < cfg["min_free_gpu_mb"]:
    raise RuntimeError(f"GPU has only {free} MiB free; not launching")


def install_plan(path: Path, plan: dict[str, Any]) -> None:
  if path.exists():
    if read_json(path) != plan:
      raise ValueError("Existing plan differs. Use a new experiment directory")
  else:
    atomic_json(path, plan)


def run_child(command: list[str], log: Path, inherited_fd: int, seed: int) -> int:
  env = os.environ.copy()
  for key in (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
  ):
    env.pop(key, None)
  env["PYTHONHASHSEED"] = str(seed)
  env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
  with log.open("x") as stream:
    proc = subprocess.Popen(
      command,
      cwd=ROOT,
      stdout=stream,
      stderr=subprocess.STDOUT,
      env=env,
      start_new_session=True,
      pass_fds=(inherited_fd,),
    )

    def stop(signum, _frame):
      os.killpg(proc.pid, signum)

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
      return proc.wait()
    finally:
      for sig, handler in old.items():
        signal.signal(sig, handler)


def verify_completed(plan: dict[str, Any], job: dict[str, Any]) -> None:
  status = read_json(Path(job["directory"]) / "status.json")
  record = status["checkpoint"]
  path = Path(record["path"]).resolve()
  path.relative_to(Path(job["directory"]).resolve() / "attempts")
  sidecar = read_json(path.with_suffix(".pt.json"))
  expected = {**sidecar, "path": str(path)}
  if (
    status["state"] != "completed"
    or status["completed_updates"] != plan["config"]["iterations"]
    or record != expected
    or record["fingerprint"] != plan["fingerprint"]
    or record["run_id"] != job["run_id"]
    or record["training_seed"] != job["training_seed"]
    or record["completed_updates"] != plan["config"]["iterations"]
    or file_sha(path) != record["sha256"]
  ):
    raise ValueError("Completed status has missing, corrupt or mismatched artifacts")


def execute_plan(plan: dict[str, Any], resume: bool) -> None:
  output, cfg = Path(plan["output"]), plan["config"]
  guard_resources(cfg, output)
  resource = cfg["device"].replace(":", "_")
  global_lock = Path(f"/tmp/mjlab_black_matrix_{os.getuid()}_{resource}.lock")
  with lock(global_lock) as resource_fd, lock(output / ".matrix.lock"):
    install_plan(output / "plan.json", plan)
    for job in plan["jobs"]:
      if source_manifest() != plan["sources"]:
        raise RuntimeError("Source changed since planning; use a new experiment")
      directory = Path(job["directory"])
      status_file = directory / "status.json"
      if status_file.exists():
        status = read_json(status_file)
        if status["state"] == "completed":
          verify_completed(plan, job)
          print(f"SKIP completed {job['run_id']}", flush=True)
          continue
        if not resume:
          raise RuntimeError(f"{job['run_id']} already started; use --resume")
      elif (directory / "attempts").exists() and not resume:
        raise RuntimeError("Existing attempts require explicit --resume")
      guard_resources(cfg, output)
      directory.mkdir(parents=True, exist_ok=True)
      stamp = f"{time.time_ns()}"
      command = [
        sys.executable,
        "-m",
        "mjlab.scripts.train_black_seed",
        "--plan",
        str(output / "plan.json"),
        "--run-id",
        job["run_id"],
        "--execute",
        "--resource-lock-fd",
        str(resource_fd),
      ]
      if resume:
        command.append("--resume")
      code = run_child(
        command, directory / f"console_{stamp}.log", resource_fd, job["training_seed"]
      )
      if code != 0:
        raise RuntimeError(f"Stopped at {job['run_id']}, exit {code}; see console log")
      if not status_file.exists() or read_json(status_file)["state"] != "completed":
        raise RuntimeError("Child exited without a verified completed status")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--template", action="store_true")
  parser.add_argument("--write-plan", action="store_true")
  parser.add_argument("--execute", action="store_true")
  parser.add_argument("--resume", action="store_true")
  args = parser.parse_args()
  if args.template:
    if args.execute or args.write_plan:
      parser.error("--template cannot execute or install a plan")
    print(json.dumps(TEMPLATE, indent=2))
    return
  if args.config is None:
    parser.error("Provide --config or --template. Training never starts by default")
  cfg = read_json(args.config)
  validate_config(cfg)
  output = args.output or ROOT / "logs" / "black_seed_matrix" / cfg["experiment_id"]
  plan = make_plan(cfg, output, source_manifest())
  if args.execute:
    execute_plan(plan, args.resume)
  else:
    if args.resume:
      parser.error("--resume requires --execute")
    if args.write_plan:
      with lock(output / ".matrix.lock"):
        install_plan(output / "plan.json", plan)
    print(json.dumps(plan, indent=2))
    print("DRY RUN: no training, evaluation, or GPU initialization", file=sys.stderr)


if __name__ == "__main__":
  main()
