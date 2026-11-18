"""Frozen-policy Black matrix evaluation with held-out selection/final seeds.

No training or evaluation is launched without --execute. Freeze every cell's
selected checkpoint before opening the final evaluation seed set.
"""

import argparse
import itertools
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.black_seed_matrix import (
  apply_overrides,
  atomic_json,
  digest,
  file_sha,
  guard_resources,
  lock,
  read_json,
  source_manifest,
  verify_completed,
)
from mjlab.scripts.train_black_seed import checkpoint_records
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.black_actuation.actuation import (
  BlackDelayedJointPositionAction,
)
from mjlab.tasks.velocity.config.black_teacher.omni_command import (
  OmniVelocityCommand,
  OmniVelocityCommandCfg,
)


def scenarios(speed: dict | None = None):
  out = []
  for name, axis, sign in [
    ("forward", 0, 1),
    ("reverse", 0, -1),
    ("turn_left", 2, 1),
    ("turn_right", 2, -1),
    ("side_left", 1, 1),
    ("side_right", 1, -1),
  ]:
    values: tuple[float, ...] = (0.1, 0.3, 0.5) if axis == 2 else (0.2, 0.5, 1.0)
    if speed is not None:
      lo, hi = speed["min_speeds"][axis], speed["max_speeds"][axis]
      values = tuple(sorted({lo, (lo + hi) / 2, hi}))
    for magnitude in values:
      cmd = [0.0, 0.0, 0.0]
      cmd[axis] = magnitude * sign
      out.append({"name": f"{name}_{magnitude}", "family": name, "command": cmd})
  combinations: list[tuple[str, tuple[float, ...]]] = [
    ("moderate", (0.5, 0.5, 0.3)),
    ("boundary", (1.0, 1.0, 0.5)),
  ]
  if speed is not None:
    combinations = [
      (
        "moderate",
        tuple(
          (lo + hi) / 2
          for lo, hi in zip(speed["min_speeds"], speed["max_speeds"], strict=True)
        ),
      ),
      ("boundary", tuple(speed["max_speeds"])),
    ]
  for scale, speeds in combinations:
    for signs in itertools.product((-1, 1), repeat=3):
      cmd = [v * s for v, s in zip(speeds, signs, strict=True)]
      out.append(
        {"name": f"mixed_{scale}_{signs}", "family": f"mixed_{scale}", "command": cmd}
      )
  out.append({"name": "standing", "family": "standing", "command": [0.0] * 3})
  return out


def sha(path):
  return file_sha(path)


def current_observations(env) -> TensorDict:
  """Recompute changed command observations; never reuse the previous-step cache."""
  return TensorDict(
    {
      name: env.observation_manager.compute_group(name, update_history=False)
      for name in env.cfg.observations
    },
    batch_size=[env.num_envs],
  )


def evaluate(args, seed, specs):
  cfg = load_env_cfg(args.job["task"], play=not args.pushes)
  rl = load_rl_cfg(args.job["task"])
  apply_overrides(cfg, rl, args.job)
  cfg.seed = seed
  cfg.scene.num_envs = len(specs) * args.replicates
  cfg.episode_length_s = 1000.0
  cfg.auto_reset = False
  cfg.commands["twist"].resampling_time_range = (1000.0, 1000.0)
  assert isinstance(cfg.commands["twist"], OmniVelocityCommandCfg)
  cfg.commands["twist"].init_velocity_prob = 0.0
  cfg.commands["twist"].debug_vis = False
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  wrap = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
  try:
    runner_cls = load_runner_cls(args.job["task"])
    assert runner_cls is not None
    runner = runner_cls(wrap, asdict(rl), device=args.device)
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    actor_before = {
      k: v.detach().clone() for k, v in runner.alg.actor.state_dict().items()
    }
    cmdterm = env.command_manager.get_term("twist")
    assert isinstance(cmdterm, OmniVelocityCommand)
    robot = env.scene["robot"]
    target = torch.tensor(
      [s["command"] for s in specs], dtype=torch.float32, device=args.device
    ).repeat_interleave(args.replicates, dim=0)
    n = env.num_envs
    fallen = torch.zeros(n, dtype=torch.bool, device=args.device)
    fall_counts = torch.zeros(n, dtype=torch.int64, device=args.device)
    dt = env.step_dt
    durations = [
      ("settle", 2.0, 0.0),
      ("command", 6.0, 1.0),
      ("reverse", 6.0, -1.0),
      ("stop", 5.0, 0.0),
    ]
    if args.smoke:
      durations = [(name, 0.04, sign) for name, _, sign in durations]
    records = {}
    action_term = env.action_manager.get_term("joint_pos")
    assert isinstance(action_term, BlackDelayedJointPositionAction)
    lags = action_term.control_delay.lags.detach().cpu().numpy().copy()
    # This teacher has single-frame observations, without observation delays.
    # Refuse future incompatible changes rather than advancing history twice.
    for group in cfg.observations.values():
      assert group.history_length in (None, 0)
      for term in group.terms.values():
        assert term.delay_max_lag == 0 and term.history_length == 0
    for phase, duration, sign in durations:
      goal = target * sign
      velocities, valid, errors, stable, heights = [], [], [], [], []
      for _ in range(round(duration / dt)):
        cmdterm.vel_command_b.copy_(goal)
        cmdterm.is_standing_env[:] = False
        cmdterm.is_heading_env[:] = False
        cmdterm.is_world_env[:] = False
        cmdterm.is_forward_env[:] = False
        cmdterm.time_left[:] = 1000.0
        obs = current_observations(env)
        with torch.no_grad():
          action = policy(obs)
          _, reward, terminated, truncated, _ = env.step(action)
        assert torch.isfinite(action).all() and torch.isfinite(reward).all(), (
          "Non-finite policy output/reward"
        )
        velocity = (
          torch.cat(
            (
              robot.data.root_link_lin_vel_b[:, :2],
              robot.data.root_link_ang_vel_b[:, 2:3],
            ),
            dim=1,
          )
          .detach()
          .clone()
        )
        assert torch.isfinite(velocity).all(), "Non-finite state"
        fallen |= terminated
        fall_counts += terminated.long()
        velocities.append(velocity.cpu().numpy())
        valid.append((~fallen).cpu().numpy().copy())
        errors.append((velocity - goal).abs().cpu().numpy())
        stable.append(
          (
            (velocity[:, :2].norm(dim=1) < 0.10)
            & (velocity[:, 2].abs() < 0.15)
            & (~fallen)
          )
          .cpu()
          .numpy()
        )
        heights.append(robot.data.root_link_pos_w[:, 2].detach().cpu().numpy().copy())
        done_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
        if len(done_ids):
          env.reset(env_ids=done_ids)
      records[phase] = {
        "velocity": np.stack(velocities),
        "valid": np.stack(valid),
        "error": np.stack(errors),
        "stable": np.stack(stable),
        "height": np.stack(heights),
      }
      print(
        f"EVAL seed={seed} phase={phase} fallen_trials={int(fallen.sum())}/{n}",
        flush=True,
      )
    unchanged = all(
      torch.equal(actor_before[k], v) for k, v in runner.alg.actor.state_dict().items()
    )
    assert unchanged, "Actor or normalization changed during evaluation"
    result = []
    final_fallen = fallen.cpu().numpy()
    for i, spec in enumerate(specs):
      sl = slice(i * args.replicates, (i + 1) * args.replicates)
      row = {
        **spec,
        "target_command": spec["command"],
        "seed": seed,
        "replicates": args.replicates,
        "trials_with_fall": int(final_fallen[sl].sum()),
        "fall_events": int(fall_counts[sl].sum()),
        "delay_counts": np.bincount(lags[sl], minlength=3).tolist(),
      }
      for phase in ("command", "reverse"):
        rec = records[phase]
        # Exclude the first 2 seconds of each commanded phase as transient.
        window = slice(round(2.0 / dt), None)
        v = rec["velocity"][window, sl]
        validmask = rec["valid"][window, sl]
        e = rec["error"][window, sl]
        row[phase] = {
          "mean_velocity": v[validmask].mean(axis=0).tolist()
          if validmask.any()
          else None,
          "mae_xyz": e[validmask].mean(axis=0).tolist() if validmask.any() else None,
          "p95_abs_error_xyz": np.quantile(e[validmask], 0.95, axis=0).tolist()
          if validmask.any()
          else None,
          "valid_samples": int(validmask.sum()),
          "total_samples": int(validmask.size),
        }
      st = records["stop"]["stable"][:, sl]
      stop_times = []
      continuous = round(0.5 / dt)
      for j in range(args.replicates):
        times = [
          k for k in range(len(st) - continuous + 1) if st[k : k + continuous, j].all()
        ]
        stop_times.append(round(times[0] * dt, 4) if times else None)
      row["stop"] = {
        "settle_time_s_each": stop_times,
        "stable_within_3s": sum(t is not None and t <= 3.0 for t in stop_times),
        "final_1s_stable_fraction": float(st[-round(1.0 / dt) :].mean()),
        "final_1s_mean_abs_velocity": np.abs(
          records["stop"]["velocity"][-round(1.0 / dt) :, sl]
        )
        .mean(axis=(0, 1))
        .tolist(),
      }
      result.append(row)
    np.savez_compressed(
      args.output.with_name(f"{args.output.stem}_seed{seed}_raw.npz"),
      **{
        f"{phase}_{key}": value
        for phase, record in records.items()
        for key, value in record.items()
      },
    )
    return {
      "seed": seed,
      "num_envs": n,
      "dt": dt,
      "actor_and_normalizer_unchanged": unchanged,
      "trials_with_fall": int(final_fallen.sum()),
      "rows": result,
    }
  finally:
    env.close()


def verified_checkpoint(plan: dict, job: dict, path: Path) -> dict:
  path = path.resolve()
  records = checkpoint_records(
    Path(job["directory"]), plan["fingerprint"], job["run_id"]
  )
  found = [record for record in records if Path(record["path"]).resolve() == path]
  if len(found) != 1:
    raise ValueError("Checkpoint is not a verified checkpoint from this training cell")
  return found[0]


def result_path(job: dict, stage: str, record: dict) -> Path:
  name = f"update_{record['completed_updates']}_{record['sha256'][:16]}.json"
  return Path(job["directory"]) / "evaluation" / stage / name


def freeze_selection(plan: dict, mapping_path: Path, execute: bool) -> None:
  mapping = read_json(mapping_path)
  if set(mapping) != {job["run_id"] for job in plan["jobs"]}:
    raise ValueError("Freeze exactly one checkpoint for EVERY training cell")
  entries = {}
  for job in plan["jobs"]:
    verify_completed(plan, job)
    state = read_json(Path(job["directory"]) / "status.json")
    if state["state"] != "completed":
      raise ValueError("Complete every training cell before freezing selection")
    record = verified_checkpoint(plan, job, Path(mapping[job["run_id"]]))
    selection_file = result_path(job, "selection", record)
    report = read_json(selection_file)
    if not report.get("complete") or not report.get("checkpoint_unchanged"):
      raise ValueError("Every selected checkpoint needs complete selection evaluation")
    if (
      report["evaluation_seeds"] != plan["config"]["selection_seeds"]
      or report["fingerprint"] != plan["fingerprint"]
      or report["run_id"] != job["run_id"]
      or report["stage"] != "selection"
      or report["checkpoint"] != record
    ):
      raise ValueError("Selection seed mismatch")
    entries[job["run_id"]] = {
      "checkpoint": record,
      "selection_report_sha256": file_sha(selection_file),
    }
  frozen = {
    "fingerprint": plan["fingerprint"],
    "entries": entries,
    "selection_rule": plan["config"]["selection_rule"],
  }
  destination = Path(plan["output"]) / "selection_frozen.json"
  if execute:
    with lock(Path(plan["output"]) / ".evaluation.lock"):
      if destination.exists():
        if read_json(destination) != frozen:
          raise ValueError("Selection is already frozen; cannot replace it")
      else:
        atomic_json(destination, frozen)
  print(json.dumps({"dry_run": not execute, "freeze": frozen}, indent=2))


def run_evaluation(plan: dict, job: dict, record: dict, stage: str) -> None:
  cfg = plan["config"]
  guard_resources(cfg, Path(plan["output"]))
  resource = cfg["device"].replace(":", "_")
  global_lock = Path(f"/tmp/mjlab_black_matrix_{os.getuid()}_{resource}.lock")
  with lock(global_lock), lock(Path(plan["output"]) / ".evaluation.lock"):
    guard_resources(cfg, Path(plan["output"]))
    frozen_path = Path(plan["output"]) / "selection_frozen.json"
    if stage == "selection" and frozen_path.exists():
      raise ValueError("Selection is frozen; do not tune after viewing final seeds")
    if stage == "final":
      frozen = read_json(frozen_path)
      if frozen["fingerprint"] != plan["fingerprint"]:
        raise ValueError("Frozen selection belongs to a different plan")
      if frozen["entries"][job["run_id"]]["checkpoint"] != record:
        raise ValueError("Final checkpoint must match frozen selection")
    output = result_path(job, stage, record)
    if output.exists():
      raise ValueError("Evaluation report already exists; it will not be overwritten")
    # Every failed/interrupted attempt has its own directory; retry does not
    # overwrite partial results and no report is published as complete early.
    attempt = output.parent / f"attempt_{time.time_ns()}"
    attempt.mkdir(parents=True, exist_ok=False)
    device = cfg["device"]
    if device.startswith("cuda:"):
      os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
      os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
      device = "cuda:0"
    else:
      os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = SimpleNamespace(
      job=job,
      checkpoint=Path(record["path"]),
      output=attempt / "metrics.json",
      device=device,
      replicates=cfg["eval_replicates"],
      pushes=cfg["eval_pushes"],
      smoke=False,
    )
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    report: dict[str, Any] = {
      "fingerprint": plan["fingerprint"],
      "stage": stage,
      "run_id": job["run_id"],
      "training_seed": job["training_seed"],
      "checkpoint": record,
      "evaluation_seeds": cfg[f"{stage}_seeds"],
      "selection_rule": cfg["selection_rule"],
      "task": job["task"],
      "scenarios": scenarios(job["overrides"].get("speed")),
      "pushes": cfg["eval_pushes"],
      "replicates": cfg["eval_replicates"],
      "protocol": "2s stand, 6s command, 6s reversal, 5s stop; tracking after 2s; "
      "fallen trials excluded from tracking means but retained as failures",
      "scope": "Evaluation seeds and parallel envs are not independent trained policies",
      "complete": False,
      "runs": [],
    }
    for seed in report["evaluation_seeds"]:
      report["runs"].append(evaluate(args, seed, report["scenarios"]))
      atomic_json(args.output, report)
    report["checkpoint_unchanged"] = file_sha(args.checkpoint) == record["sha256"]
    if not report["checkpoint_unchanged"]:
      raise RuntimeError("Checkpoint changed during evaluation")
    report["complete"] = True
    atomic_json(args.output, report)
    atomic_json(output, report)
    print(f"EVAL_COMPLETE {output}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--run-id")
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--stage", choices=("selection", "final"), default="selection")
  parser.add_argument("--freeze-selection", type=Path)
  parser.add_argument("--execute", action="store_true")
  args = parser.parse_args()
  plan = read_json(args.plan)
  payload = {key: plan[key] for key in ("config", "sources", "jobs", "task")}
  if digest(payload) != plan["fingerprint"] or source_manifest() != plan["sources"]:
    raise ValueError("Plan/source fingerprint mismatch")
  if args.freeze_selection:
    freeze_selection(plan, args.freeze_selection, args.execute)
    return
  if not args.run_id or not args.checkpoint:
    parser.error("Provide --run-id and --checkpoint, or --freeze-selection")
  jobs = [job for job in plan["jobs"] if job["run_id"] == args.run_id]
  if len(jobs) != 1:
    raise ValueError("Unknown run_id")
  job = jobs[0]
  record = verified_checkpoint(plan, job, args.checkpoint)
  if args.execute:
    run_evaluation(plan, job, record, args.stage)
  else:
    print(
      json.dumps(
        {
          "dry_run": True,
          "run_id": args.run_id,
          "stage": args.stage,
          "checkpoint": record,
          "seeds": plan["config"][f"{args.stage}_seeds"],
        },
        indent=2,
      )
    )


if __name__ == "__main__":
  main()
