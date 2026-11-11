"""Independent continuation validation. No training, reward edits, or label fitting.

New in-distribution command/randomization draws are not an unseen physical domain.
Final holdout is report-only: never use its scores to select continuation checkpoints.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mjlab.scripts.black_seed_matrix import TASK, atomic_json, digest, file_sha
from mjlab.scripts.evaluate_black_candidates import evaluate

VALIDATION_SEEDS = (921101, 921102)
FINAL_SEEDS = (931101, 931102)
REPLICATES = 8


def continuation_scenarios(split="validation"):
  if split not in ("validation", "final"):
    raise ValueError("Unknown split")
  cases = []

  def add(family, command):
    cases.append(
      dict(name=f"{family}_{len(cases):02d}", family=family, command=command)
    )

  # Final command values are fixed independently, not adjusted after validation.
  small, large = (0.07, 0.12) if split == "validation" else (0.09, 0.14)
  for axis in range(3):
    for magnitude in (small, 0.23 if axis == 2 else large):
      for sign in (-1, 1):
        command = [0.0, 0.0, 0.0]
        command[axis] = sign * magnitude
        add("low", command)
  scale = 1.0 if split == "validation" else 0.9
  for command in (
    [0.12, -0.23, 0.07],
    [-0.23, 0.12, 0.23],
    [0.07, 0.23, -0.12],
    [-0.12, -0.07, -0.23],
  ):
    add("low", [v * scale for v in command])
  for command in ([-0.85, 0, 0], [0, -0.8, 0], [0, 0, 0.45], [0.65, 0.5, -0.4]):
    add("retention", [v * scale for v in command])
  for command in ([0.97, 0, 0], [1.0, -1.0, 0.5], [-1.0, 1.0, -0.5], [1.0, 1.0, 0.5]):
    add("high", command)
    cases[-1]["previously_seen_boundary_regression"] = command[0] != 0.97
  return cases


def aggregate_runs(runs):
  """MAE is survivor-only; falls and coverage must always be read alongside it."""
  result = {}
  for mode in ("all", "abrupt", "ramp"):
    selected = [r for r in runs if mode == "all" or r["mode"] == mode]
    result[mode] = {}
    for family in ("all", "low", "retention", "high"):
      rows = [
        row
        for run in selected
        for row in run["rows"]
        if family == "all" or row["family"] == family
      ]
      trials = sum(row["replicates"] for row in rows)
      errors, ratios, reversals, stops = [], [], [], []
      valid = total = 0
      for row in rows:
        for phase in ("command", "reverse"):
          p = row[phase]
          valid += p["valid_samples"]
          total += p["total_samples"]
          if p["mae_xyz"] is not None:
            errors.append((p["mae_xyz"], p["valid_samples"]))
          ratios.extend(v for v in p["response_ratio_xyz"] if v is not None)
        reversals.extend(row["reversal_time_s_each"])
        stops.extend(row["stop"]["settle_time_s_each"])
      weight = sum(n for _, n in errors)
      xyz = (
        [sum(e[a] * n for e, n in errors) / weight for a in range(3)]
        if weight
        else None
      )
      result[mode][family] = {
        "trials": trials,
        "fall_fraction": sum(r["trials_with_fall"] for r in rows) / trials
        if trials
        else None,
        "tracking_mae_xyz": xyz,
        "tracking_mae": sum(xyz) / 3 if xyz is not None else None,
        "valid_sample_fraction": valid / total if total else 0.0,
        "response_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "reversal_success_fraction": sum(t is not None for t in reversals)
        / len(reversals)
        if reversals
        else 0.0,
        "stop_within_3s_fraction": sum(t is not None and t <= 3 for t in stops)
        / len(stops)
        if stops
        else 0.0,
      }
  return result


def protocol(split, smoke):
  return {
    "schema_version": 1,
    "task": TASK,
    "split": split,
    "seeds": list(VALIDATION_SEEDS if split == "validation" else FINAL_SEEDS),
    "replicates": REPLICATES,
    "modes": ["abrupt", "ramp"],
    "smoke": smoke,
    "training_randomization_and_pushes": True,
    "scenarios": continuation_scenarios(split),
    "scope": "In-distribution new draws plus labeled previously seen boundary regressions; not an unseen physical domain",
    "tracking": "Survivor-only terminal-goal MAE after 3s; read fall/coverage too",
    "high_teacher": "Disabled unless independently verified better than Omni forward",
  }


def evaluate_checkpoint(
  checkpoint, output, device="cpu", split="validation", smoke=False
):
  checkpoint, output = Path(checkpoint), Path(output)
  if output.exists():
    raise FileExistsError(f"Refusing to overwrite {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  sha = file_sha(checkpoint)
  spec = protocol(split, smoke)
  report: dict[str, Any] = dict(
    protocol=spec,
    protocol_sha256=digest(spec),
    complete=False,
    checkpoint=dict(path=str(checkpoint.resolve()), sha256=sha),
    runs=[],
  )
  # Baseline evaluator runs inference under no_grad and verifies actor + normalizer.
  for mode in spec["modes"]:
    for seed in spec["seeds"]:
      if file_sha(checkpoint) != sha:
        raise RuntimeError("Checkpoint changed before evaluation")
      args = SimpleNamespace(
        checkpoint=checkpoint,
        output=output.with_name(f"{output.stem}_{mode}.json"),
        device=device,
        replicates=REPLICATES,
        pushes=True,
        smoke=smoke,
        mode=mode,
      )
      result = evaluate(args, seed, spec["scenarios"])
      report["runs"].append(dict(mode=mode, **result))
  if file_sha(checkpoint) != sha:
    raise RuntimeError("Checkpoint changed during evaluation")
  report.update(
    complete=True, checkpoint_unchanged=True, metrics=aggregate_runs(report["runs"])
  )
  # Reject nonstandard JSON nonfinites, while smoke's empty tracking is explicit null.
  json.dumps(report, allow_nan=False)
  atomic_json(output, report)
  return report


def _verified(report):
  p = report.get("protocol", {})
  expected = protocol("validation", False)
  if not (
    report.get("complete")
    and report.get("checkpoint_unchanged")
    and p == expected
    and report.get("protocol_sha256") == digest(expected)
  ):
    return False
  runs = report.get("runs", [])
  if len(runs) != 4 or {(r["mode"], r["seed"]) for r in runs} != {
    (m, s) for m in expected["modes"] for s in VALIDATION_SEEDS
  }:
    return False
  if not all(r.get("actor_and_normalizer_unchanged") for r in runs):
    return False
  try:
    json.dumps(report, allow_nan=False)
    if report["metrics"] != aggregate_runs(runs):
      return False
    for mode in ("all", "abrupt", "ramp"):
      for family in ("all", "low", "retention", "high"):
        metric = report["metrics"][mode][family]
        for key in ("tracking_mae", "fall_fraction", "response_ratio_mean"):
          if metric[key] is None or not math.isfinite(metric[key]):
            return False
  except (KeyError, TypeError, ValueError):
    return False
  return True


def _degradation(reference, current, severe, include_low=False):
  reasons = []
  # Compare both modes independently so ramp cannot conceal abrupt regressions.
  for mode in ("all", "abrupt", "ramp"):
    for family in (
      ("retention", "high")
      if severe and not include_low
      else ("low", "retention", "high")
    ):
      ref, cur = (r["metrics"][mode][family] for r in (reference, current))
      fall_limit, relative, absolute = (
        (0.10, 0.35, 0.01) if severe else (0.03, 0.15, 0.01)
      )
      if cur["fall_fraction"] - ref["fall_fraction"] > fall_limit + 1e-12:
        reasons.append(f"{mode}/{family}: fall regression")
      if (
        cur["tracking_mae"] - ref["tracking_mae"]
        > max(absolute, relative * ref["tracking_mae"]) + 1e-12
      ):
        reasons.append(f"{mode}/{family}: tracking regression")
  return reasons


def gate_checkpoint(reference, current, previous=None, stage1_reference=None):
  """Reference is frozen Omni. Previous is the immediately preceding validation.

  Full validation, not smoke or final holdout, is eligible for stage transition.
  Two successive meaningful regressions on the same metric trigger early stopping.
  No low-speed improvement prerequisite is imposed on safe stage-2 continuation.
  """
  if not _verified(reference) or not _verified(current):
    return dict(
      eligible=False, stop=True, reasons=["Unverified/incomplete/nonfinite validation"]
    )
  reasons = _degradation(reference, current, severe=True)
  if stage1_reference is not None:
    if not _verified(stage1_reference):
      return dict(eligible=False, stop=True, reasons=["Unverified stage-1 reference"])
    reasons += [
      f"Stage-1 retention: {r}"
      for r in _degradation(stage1_reference, current, True, include_low=True)
    ]
  repeated = []
  if previous is not None and _verified(previous):
    repeated = sorted(
      set(_degradation(reference, previous, False))
      & set(_degradation(reference, current, False))
    )
  if stage1_reference is not None and previous is not None and _verified(previous):
    repeated += sorted(
      set(_degradation(stage1_reference, previous, False))
      & set(_degradation(stage1_reference, current, False))
    )
  return dict(
    eligible=not (reasons or repeated),
    stop=bool(reasons or repeated),
    reasons=reasons + [f"Repeated: {r}" for r in repeated],
  )


def select_eligible_checkpoint(reference, validations, stage1_reference=None):
  """Latest eligible checkpoint before first stop; never consult final holdout."""
  selected, previous = None, None
  for current in validations:
    decision = gate_checkpoint(reference, current, previous, stage1_reference)
    if decision["stop"]:
      break
    if decision["eligible"]:
      selected = current["checkpoint"]
    previous = current
  return selected


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--split", choices=("validation", "final"), default="validation")
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--execute", action="store_true")
  args = parser.parse_args()
  if not args.execute:
    print(
      json.dumps(
        dict(dry_run=True, protocol=protocol(args.split, args.smoke)), indent=2
      )
    )
    return
  evaluate_checkpoint(args.checkpoint, args.output, args.device, args.split, args.smoke)
  print(f"EVAL_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
  main()
