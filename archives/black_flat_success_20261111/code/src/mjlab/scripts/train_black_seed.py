"""One isolated PPO policy in a Black matrix. No training without --execute."""

import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

from mjlab.scripts.black_seed_matrix import (
  RESUME_SCOPE,
  apply_overrides,
  atomic_json,
  digest,
  file_sha,
  guard_resources,
  lock,
  read_json,
  source_manifest,
  validate_config,
)


def checkpoint_records(directory: Path, fingerprint: str, run_id: str) -> list[dict]:
  records = []
  for sidecar in directory.glob("attempts/*/model_*.pt.json"):
    record = read_json(sidecar)
    path = sidecar.with_suffix("")
    if record["fingerprint"] != fingerprint or record["run_id"] != run_id:
      raise ValueError("Checkpoint belongs to another run/plan")
    if path.is_file() and file_sha(path) == record["sha256"]:
      records.append({**record, "path": str(path)})
  return sorted(
    records, key=lambda record: (record["completed_updates"], record["path"])
  )


def resume_start(
  record: dict, info: dict, loaded_iteration: int, expected: dict
) -> int:
  """Validate checkpoint ownership and convert last index to completed count."""
  completed = record["completed_updates"]
  metadata = {**expected, "completed_updates": completed}
  if info.get("black_matrix") != metadata or loaded_iteration + 1 != completed:
    raise ValueError("Checkpoint iteration or ownership metadata mismatch")
  return completed


def train(plan_path: Path, run_id: str, resume: bool) -> None:
  plan = read_json(plan_path)
  cfg = plan["config"]
  validate_config(cfg)
  if cfg["device"].startswith("cuda:"):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["device"].split(":")[1]
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  # Heavy dependencies are intentionally imported only after explicit execution.
  import numpy as np
  import torch

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends

  plan = read_json(plan_path)
  payload = {key: plan[key] for key in ("config", "sources", "jobs", "task")}
  if digest(payload) != plan["fingerprint"] or source_manifest() != plan["sources"]:
    raise ValueError("Plan/source fingerprint mismatch")
  jobs = [job for job in plan["jobs"] if job["run_id"] == run_id]
  if len(jobs) != 1:
    raise ValueError("Unknown run_id")
  job, cfg = jobs[0], plan["config"]
  validate_config(cfg)
  runtime = {
    "python": platform.python_version(),
    **{
      name: importlib.metadata.version(name)
      for name in ("torch", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")
    },
  }
  if runtime["rsl-rl-lib"] != "5.5.1":
    raise ValueError("This continuation adapter requires rsl-rl-lib 5.5.1")
  if os.environ.get("PYTHONHASHSEED") != str(job["training_seed"]):
    raise ValueError("Launch through black_seed_matrix to set PYTHONHASHSEED")
  directory = Path(job["directory"])
  with lock(directory / ".run.lock"):
    state_path = directory / "status.json"
    if state_path.exists() and read_json(state_path)["state"] == "completed":
      raise ValueError("Run is already complete")
    records = checkpoint_records(directory, plan["fingerprint"], run_id)
    previous = records[-1] if records else None
    if (state_path.exists() or (directory / "attempts").exists()) and not resume:
      raise ValueError("Existing run requires --resume; no implicit warm start")
    if resume and previous is None and (directory / "attempts").exists():
      raise ValueError("No verified checkpoint. Use a new experiment for a fresh run")
    if previous and previous["runtime"] != runtime:
      raise ValueError("Runtime versions changed; refuse incompatible resume")
    start = previous["completed_updates"] if previous else 0
    remaining = cfg["iterations"] - start
    if remaining <= 0:
      atomic_json(
        state_path,
        {
          "state": "completed",
          "completed_updates": start,
          "checkpoint": previous,
          "resume_scope": RESUME_SCOPE,
        },
      )
      return
    attempt = directory / "attempts" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    seed = job["training_seed"]
    # Resume deliberately resets RNG and physics. This is not trajectory replay.
    random.seed(seed)
    np.random.seed(seed)
    device = cfg["device"]
    if device.startswith("cuda:"):
      physical = device.split(":")[1]
      os.environ["CUDA_VISIBLE_DEVICES"] = physical
      os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
      device = "cuda:0"
      torch.cuda.manual_seed_all(seed)
    else:
      os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.manual_seed(seed)
    configure_torch_backends()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    env_cfg, agent_cfg = load_env_cfg(job["task"]), load_rl_cfg(job["task"])
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    env_cfg.seed = agent_cfg.seed = seed
    env_cfg.scene.num_envs = cfg["num_envs"]
    env_cfg.commands["twist"].debug_vis = False
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    agent_cfg.resume = False
    agent_cfg.max_iterations = cfg["iterations"]
    agent_cfg.save_interval = cfg["save_interval"]
    apply_overrides(env_cfg, agent_cfg, job)
    dump_yaml(attempt / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(attempt / "params" / "agent.yaml", asdict(agent_cfg))
    atomic_json(
      attempt / "identity.json",
      {
        **job,
        "fingerprint": plan["fingerprint"],
        "resume_from": previous,
        "resume_scope": RESUME_SCOPE,
        "target_updates": cfg["iterations"],
        "runtime": runtime,
      },
    )

    class AtomicSeedRunner(VelocityOnPolicyRunner):
      def save(self, path: str, infos=None):
        # A nonfinite learned policy is an actual training failure, never a
        # checkpoint eligible for evaluation or a trigger for the next job.
        for module in (self.alg.actor, self.alg.critic):
          if any(not torch.isfinite(p).all() for p in module.parameters()):
            raise RuntimeError("Nonfinite trained parameters; stop serial run")
        target = Path(path)
        temporary = target.with_name(f".{target.stem}.partial.pt")
        completed = self.current_learning_iteration + 1
        super().save(
          str(temporary),
          {
            **(infos or {}),
            "black_matrix": {
              "run_id": run_id,
              "fingerprint": plan["fingerprint"],
              "completed_updates": completed,
              "training_seed": seed,
              "runtime": runtime,
            },
          },
        )
        os.replace(temporary, target)
        atomic_json(
          target.with_suffix(".pt.json"),
          {
            "run_id": run_id,
            "fingerprint": plan["fingerprint"],
            "completed_updates": completed,
            "training_seed": seed,
            "runtime": runtime,
            "sha256": file_sha(target),
          },
        )

    env = None
    atomic_json(
      state_path,
      {
        "state": "running",
        "pid": os.getpid(),
        "attempt": str(attempt),
        "completed_updates": start,
      },
    )
    try:
      env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
      wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
      runner = AtomicSeedRunner(wrapper, asdict(agent_cfg), str(attempt), device)
      initial_path = attempt / "initial_actor.pt"
      torch.save(runner.alg.actor.state_dict(), initial_path)
      if previous:
        info = runner.load(previous["path"], map_location=device)
        resume_start(
          previous,
          info,
          runner.current_learning_iteration,
          {
            "run_id": run_id,
            "fingerprint": plan["fingerprint"],
            "training_seed": seed,
            "runtime": runtime,
          },
        )
        # RSL's checkpoint iter is the last completed zero-based index; the
        # next learn() must start at index+1 to avoid repeating that index.
        runner.current_learning_iteration = start
      runner.learn(remaining, init_at_random_ep_len=True)
      final_records = checkpoint_records(directory, plan["fingerprint"], run_id)
      if (
        not final_records or final_records[-1]["completed_updates"] != cfg["iterations"]
      ):
        raise RuntimeError("Missing verified final checkpoint")
      atomic_json(
        state_path,
        {
          "state": "completed",
          "completed_updates": cfg["iterations"],
          "checkpoint": final_records[-1],
          "resume_scope": RESUME_SCOPE,
        },
      )
    except BaseException as error:
      atomic_json(
        state_path,
        {
          "state": "interrupted_or_failed",
          "error": f"{type(error).__name__}: {error}",
          "attempt": str(attempt),
          "resume_scope": RESUME_SCOPE,
        },
      )
      raise
    finally:
      if env is not None:
        env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--execute", action="store_true")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--resource-lock-fd", type=int, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if not args.execute:
    print(json.dumps({"dry_run": True, "plan": str(args.plan), "run_id": args.run_id}))
    return
  plan = read_json(args.plan)
  cfg = plan["config"]
  validate_config(cfg)
  resource = cfg["device"].replace(":", "_")
  global_lock = Path(f"/tmp/mjlab_black_matrix_{os.getuid()}_{resource}.lock")
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
    os.environ.pop(key, None)
  if args.resource_lock_fd is not None:
    # The scheduler passes its open lock, so it survives scheduler termination.
    held = os.fstat(args.resource_lock_fd)
    expected = global_lock.stat()
    if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
      raise ValueError("Wrong inherited resource lock")
    fcntl.flock(args.resource_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    guard_resources(cfg, Path(plan["output"]))
    train(args.plan, args.run_id, args.resume)
  else:
    with lock(global_lock):
      guard_resources(cfg, Path(plan["output"]))
      train(args.plan, args.run_id, args.resume)


if __name__ == "__main__":
  main()
