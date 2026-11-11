"""Evaluate two final candidate policies and original Omni on a frozen protocol.

No GPU work without --execute. Final model_2999 is predetermined, not selected
using holdout scores. Run only after both candidate training cells complete.
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

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.black_seed_matrix import (
  ROOT,
  TASK,
  atomic_json,
  digest,
  file_sha,
  guard_resources,
  lock,
  read_json,
  source_manifest,
  verify_completed,
)
from mjlab.scripts.evaluate_black_seed import (
  current_observations,
  scenarios,
  verified_checkpoint,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.black_actuation.actuation import (
  BlackDelayedJointPositionAction,
)
from mjlab.tasks.velocity.config.black_teacher.omni_command import (
  OmniVelocityCommand,
  OmniVelocityCommandCfg,
)


def evaluate(args, seed, specs):
  cfg = load_env_cfg(TASK, play=not args.pushes)
  rl = load_rl_cfg(TASK)
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
    runner_cls = load_runner_cls(TASK)
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
    applied = torch.zeros_like(target)
    for phase, duration, sign in durations:
      fallen_before = fallen.clone()
      goal = target * sign
      velocities, valid, errors, stable, heights, commands = [], [], [], [], [], []
      for _ in range(round(duration / dt)):
        applied = advance_command(applied, goal, dt, args.mode)
        cmdterm.vel_command_b.copy_(applied)
        commands.append(applied.cpu().numpy().copy())
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
        "command": np.stack(commands),
        "new_falls": (fallen & ~fallen_before).cpu().numpy(),
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
    result = summarize(records, specs, args.replicates, dt, fallen.cpu().numpy(), lags)
    final_fallen = fallen.cpu().numpy()
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


SEEDS = [81101, 81102]
BASELINE = (
  ROOT
  / "logs/rsl_rl/black_teacher_omni_flat/2026-11-10_19-06-42_omni_4096_3000/model_2999.pt"
)
BASELINE_SHA = "a5e5a066f9f5cb809cc7e338c2db03270c1daf3826c9f92794a875af57959629"
CANDIDATES = {
  f"Mjlab-Velocity-Flat-Black-Teacher-{s}Speed-Candidate" for s in ("Low", "High")
}


def candidate_scenarios():
  """Original 35 cases plus deduplicated small pure and low mixed commands."""
  out = scenarios()
  seen = {tuple(s["command"]) for s in out}

  def add(name, family, command):
    if tuple(command) not in seen:
      out.append(dict(name=name, family=family, command=command))
      seen.add(tuple(command))

  for axis in range(3):
    for magnitude in (0.05, 0.10, 0.15):
      for sign in (-1, 1):
        command = [0.0] * 3
        command[axis] = magnitude * sign
        add(f"low_axis{axis}_{magnitude}_{sign}", "low_pure", command)
  for signs in itertools.product((-1, 1), repeat=3):
    command = [v * s for v, s in zip((0.3, 0.4, 0.3), signs, strict=True)]
    add(f"low_mixed_{signs}", "low_mixed", command)
  return out


def advance_command(current, goal, dt, mode):
  if mode == "abrupt":
    return goal.clone()
  delta = goal - current
  xy = delta[:, :2]
  scale = (dt / xy.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
  return current + torch.cat(
    (xy * scale, delta[:, 2:3].clamp(-0.5 * dt, 0.5 * dt)), dim=1
  )


def sustained_times(mask, dt):
  """Completion time of first continuous 0.5s success, not its start."""
  width = max(1, round(0.5 / dt))
  result = []
  for j in range(mask.shape[1]):
    times = [k for k in range(len(mask) - width + 1) if mask[k : k + width, j].all()]
    result.append(round((times[0] + width) * dt, 4) if times else None)
  return result


def summarize(records, specs, replicates, dt, fallen, lags):
  rows = []
  for i, spec in enumerate(specs):
    sl = slice(i * replicates, (i + 1) * replicates)
    target = np.asarray(spec["command"])
    row = {
      **spec,
      "replicates": replicates,
      "trials_with_fall": int(fallen[sl].sum()),
      "delay_counts": np.bincount(lags[sl], minlength=3).tolist(),
      "phase_first_falls": {
        p: int(r["new_falls"][sl].sum()) for p, r in records.items()
      },
    }
    for phase, sign in [("command", 1), ("reverse", -1)]:
      rec = records[phase]
      window = slice(round(3.0 / dt), None)
      valid = rec["valid"][window, sl]
      v, error = rec["velocity"][window, sl], rec["error"][window, sl]
      mean = v[valid].mean(axis=0) if valid.any() else None
      row[phase] = {
        "tracking_window_start_s": 3.0,
        "mean_velocity": mean.tolist() if mean is not None else None,
        "mae_xyz": error[valid].mean(axis=0).tolist() if valid.any() else None,
        "p95_abs_error_xyz": np.quantile(error[valid], 0.95, axis=0).tolist()
        if valid.any()
        else None,
        "response_ratio_xyz": [
          float(mean[a] / (sign * target[a]))
          if mean is not None and target[a] != 0
          else None
          for a in range(3)
        ],
        "valid_samples": int(valid.sum()),
        "total_samples": int(valid.size),
      }
    rev = records["reverse"]
    active = np.abs(target) > 0
    crossed = ((rev["velocity"][:, sl] * (-np.sign(target))) > 0)[:, :, active].all(
      axis=2
    )
    row["reversal_time_s_each"] = (
      sustained_times(crossed & rev["valid"][:, sl], dt)
      if active.any()
      else [None] * replicates
    )
    rec = records["stop"]
    times = sustained_times(rec["stable"][:, sl], dt)
    valid = rec["valid"][-round(1 / dt) :, sl]
    v = rec["velocity"][-round(1 / dt) :, sl]
    row["stop"] = {
      "settle_time_s_each": times,
      "stable_within_3s": sum(t is not None and t <= 3 for t in times),
      "final_1s_stable_fraction": float(rec["stable"][-round(1 / dt) :, sl].mean()),
      "final_1s_mean_abs_velocity": np.abs(v[valid]).mean(axis=0).tolist()
      if valid.any()
      else None,
      "valid_samples": int(valid.sum()),
    }
    rows.append(row)
  return rows


def validate_plan(plan):
  payload = {k: plan[k] for k in ("config", "sources", "jobs", "task")}
  if digest(payload) != plan["fingerprint"] or source_manifest() != plan["sources"]:
    raise ValueError("Plan/source fingerprint mismatch")
  cfg = plan["config"]
  if (
    cfg["iterations"] != 3000
    or cfg["eval_replicates"] != 16
    or cfg["final_seeds"] != SEEDS
  ):
    raise ValueError(
      "Require fixed 3000 updates, 16 replicates and held-out seeds 81101/81102"
    )
  if len(plan["jobs"]) != 2 or {j["task"] for j in plan["jobs"]} != CANDIDATES:
    raise ValueError("Require exactly the low/high candidate training cells")
  train = [j["training_seed"] for j in plan["jobs"]]
  if len(set(train)) != 2 or set(train) & set(SEEDS):
    raise ValueError("Training seeds must be distinct and disjoint from holdout seeds")
  entries = []
  for job in plan["jobs"]:
    verify_completed(plan, job)
    state = read_json(Path(job["directory"]) / "status.json")
    record = verified_checkpoint(plan, job, Path(state["checkpoint"]["path"]))
    if (
      Path(record["path"]).name != "model_2999.pt"
      or record["completed_updates"] != 3000
    ):
      raise ValueError("Only the predetermined final model_2999 is eligible")
    entries.append(
      {"run_id": job["run_id"], "training_task": job["task"], "checkpoint": record}
    )
  if file_sha(BASELINE) != BASELINE_SHA:
    raise ValueError("Original Omni baseline hash mismatch")
  entries.append(
    {
      "run_id": "original_omni",
      "training_task": TASK,
      "checkpoint": {"path": str(BASELINE), "sha256": BASELINE_SHA},
    }
  )
  return entries


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--execute", action="store_true")
  args = parser.parse_args()
  plan = read_json(args.plan)
  entries = validate_plan(plan)
  protocol: dict[str, Any] = {
    "schema_version": 1,
    "fingerprint": plan["fingerprint"],
    "evaluation_task": TASK,
    "evaluation_seeds": SEEDS,
    "replicates": 16,
    "selection_rule": "No model selection; final model_2999 fixed before training",
    "modes": ["abrupt", "ramp"],
    "linear_accel_m_s2": 1.0,
    "yaw_accel_rad_s2": 0.5,
    "phases_s": {"settle": 2, "command": 6, "reverse": 6, "stop": 5},
    "tracking_start_s": 3,
    "sustained_success_s": 0.5,
    "fall_mask": "All subsequent samples excluded after first termination; resets only for simulation health",
    "scope": "One trained policy per candidate; evaluation replicas are not independent training seeds",
    "diagnostics_unavailable": ["foot_slip", "torque_saturation"],
    "pushes": plan["config"]["eval_pushes"],
    "scenarios": candidate_scenarios(),
    "policies": entries,
  }
  if not args.execute:
    print(json.dumps({"dry_run": True, "protocol": protocol}, indent=2))
    return
  cfg, output = plan["config"], Path(plan["output"]) / "evaluation_candidates"
  guard_resources(cfg, Path(plan["output"]))
  resource = cfg["device"].replace(":", "_")
  with (
    lock(Path(f"/tmp/mjlab_black_matrix_{os.getuid()}_{resource}.lock")),
    lock(output / ".evaluation.lock"),
  ):
    guard_resources(cfg, Path(plan["output"]))
    validate_plan(plan)
    frozen = output / "protocol_frozen.json"
    if frozen.exists() and read_json(frozen) != protocol:
      raise ValueError("Frozen evaluation protocol differs")
    if (output / "complete.json").exists():
      raise ValueError("Completed evaluation will not be overwritten")
    if not frozen.exists():
      atomic_json(frozen, protocol)
    attempt = output / f"attempt_{time.time_ns()}"
    attempt.mkdir(parents=True, exist_ok=False)
    device = cfg["device"]
    if device.startswith("cuda:"):
      os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
      os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
      device = "cuda:0"
    else:
      os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    report: dict[str, Any] = {
      **protocol,
      "complete": False,
      "attempt": str(attempt),
      "runs": [],
    }
    for index, entry in enumerate(entries):
      checkpoint = Path(entry["checkpoint"]["path"])
      for mode in protocol["modes"]:
        for seed in SEEDS:
          if file_sha(checkpoint) != entry["checkpoint"]["sha256"]:
            raise RuntimeError("Checkpoint changed before inference")
          evaluation_args = SimpleNamespace(
            checkpoint=checkpoint,
            output=attempt / f"policy{index}_{mode}.json",
            device=device,
            replicates=16,
            pushes=cfg["eval_pushes"],
            smoke=False,
            mode=mode,
          )
          result = evaluate(evaluation_args, seed, protocol["scenarios"])
          if file_sha(checkpoint) != entry["checkpoint"]["sha256"]:
            raise RuntimeError("Checkpoint changed during inference")
          report["runs"].append({"run_id": entry["run_id"], "mode": mode, **result})
          atomic_json(attempt / "metrics.json", report)
    if source_manifest() != plan["sources"]:
      raise RuntimeError("Source changed during evaluation")
    report.update(complete=True, checkpoint_unchanged=True)
    atomic_json(attempt / "metrics.json", report)
    atomic_json(output / "complete.json", report)
    print(f"EVAL_COMPLETE {output / 'complete.json'}", flush=True)


if __name__ == "__main__":
  main()
