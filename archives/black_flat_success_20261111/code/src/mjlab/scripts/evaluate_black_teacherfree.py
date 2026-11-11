"""Independent teacher-free continuation validation from selected +1000 source.

One policy, no teachers. Frozen Omni is the reference, selected +1000 is the
retention anchor. Stopping tolerances are not convergence proof.

New in-distribution command/randomization draws are not an unseen physical domain.
Final holdout is report-only: never use its scores to select continuation checkpoints.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from mjlab.scripts.black_seed_matrix import TASK, atomic_json, digest, file_sha
from mjlab.scripts.evaluate_black_candidates import evaluate

VALIDATION_SEEDS = (941101, 941102)
FINAL_SEEDS = (951101, 951102)
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
  for command in (
    [-0.85, 0, 0],
    [0.85, 0, 0],
    [0, -0.8, 0],
    [0, 0.8, 0],
    [0, 0, -0.45],
    [0, 0, 0.45],
    [0.65, 0.5, -0.4],
    [-0.65, -0.5, 0.4],
  ):
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
    "experiment": "teacherfree_from_selected_plus1000",
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
    "teachers": "None; a single policy is evaluated",
    "phases_s": {"settle": 2, "command": 6, "reverse": 6, "stop": 5},
    "selection": "Validation only; final is paired report-only against frozen Omni",
    "interpretation": "Stopping tolerances are not convergence or nonforgetting proof",
  }


def verify_finite_checkpoint(checkpoint):
  """CPU-only actor/critic preflight; ignore obsolete training/teacher metadata."""
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  for key in ("actor_state_dict", "critic_state_dict"):
    state = payload.get(key)
    if not isinstance(state, dict) or not state:
      raise ValueError(f"Missing or empty {key}")
    for name, value in state.items():
      if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise FloatingPointError(f"Invalid/nonfinite checkpoint tensor: {key}/{name}")


def evaluate_checkpoint(
  checkpoint, output, device="cpu", split="validation", smoke=False
):
  spec = protocol(split, smoke)
  checkpoint, output = Path(checkpoint), Path(output)
  if output.exists():
    raise FileExistsError(f"Refusing to overwrite {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  sha = file_sha(checkpoint)
  verify_finite_checkpoint(checkpoint)
  report: dict[str, Any] = dict(
    protocol=spec,
    protocol_sha256=digest(spec),
    complete=False,
    checkpoint_tensors_finite=True,
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
      if not result.get("actor_and_normalizer_unchanged"):
        raise RuntimeError("Actor or normalization changed during evaluation")
      if len(result["rows"]) != len(spec["scenarios"]):
        raise RuntimeError("Incomplete scenario results")
      for row, scenario in zip(result["rows"], spec["scenarios"], strict=True):
        if row["name"] != scenario["name"] or row["family"] != scenario["family"]:
          raise RuntimeError("Scenario identity mismatch")
        # Frozen rollout's command key holds metrics, not the target vector.
        row["target_command"] = list(scenario["command"])
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


def _verified(report, split="validation", smoke=False):
  """Validate provenance and coverage; optional split is for paired reporting.

  Gates use validation/non-smoke defaults. Smoke is never verified.
  """
  if smoke:
    return False
  try:
    expected = protocol(split, False)
    if not (
      report.get("complete") is True
      and report.get("checkpoint_tensors_finite") is True
      and report.get("checkpoint_unchanged") is True
      and report.get("protocol") == expected
      and report.get("protocol_sha256") == digest(expected)
    ):
      return False
    checkpoint = report["checkpoint"]
    if not checkpoint["path"] or len(checkpoint["sha256"]) != 64:
      return False
    int(checkpoint["sha256"], 16)
    runs = report["runs"]
    if len(runs) != 4 or {(r["mode"], r["seed"]) for r in runs} != {
      (m, seed) for m in expected["modes"] for seed in expected["seeds"]
    }:
      return False
    for run in runs:
      if run.get("actor_and_normalizer_unchanged") is not True:
        return False
      if len(run["rows"]) != len(expected["scenarios"]):
        return False
      if run["num_envs"] != REPLICATES * len(expected["scenarios"]):
        return False
      for row, scenario in zip(run["rows"], expected["scenarios"], strict=True):
        if any(row[key] != scenario[key] for key in ("name", "family")):
          return False
        if row["target_command"] != scenario["command"]:
          return False
        if row.get("previously_seen_boundary_regression") != scenario.get(
          "previously_seen_boundary_regression"
        ):
          return False
        if row["replicates"] != REPLICATES or not (
          0 <= row["trials_with_fall"] <= REPLICATES
        ):
          return False
        for phase in ("command", "reverse"):
          p = row[phase]
          if (
            not 0 <= p["valid_samples"] <= p["total_samples"] or p["total_samples"] <= 0
          ):
            return False
          if p["mae_xyz"] is not None and (
            len(p["mae_xyz"]) != 3 or any(v < 0 for v in p["mae_xyz"])
          ):
            return False
          if len(p["response_ratio_xyz"]) != 3:
            return False
        for times in (row["reversal_time_s_each"], row["stop"]["settle_time_s_each"]):
          if len(times) != REPLICATES or any(t is not None and t < 0 for t in times):
            return False
      if run["trials_with_fall"] != sum(r["trials_with_fall"] for r in run["rows"]):
        return False
    json.dumps(report, allow_nan=False)
    if report["metrics"] != aggregate_runs(runs):
      return False
    for mode in ("all", "abrupt", "ramp"):
      for family in ("all", "low", "retention", "high"):
        metric = report["metrics"][mode][family]
        for key in ("tracking_mae", "fall_fraction", "response_ratio_mean"):
          if metric[key] is None or not math.isfinite(metric[key]):
            return False
  except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
    return False
  return True


def paired_final_report(reference, selected):
  """Report frozen Omni and selected single policy on identical final draws.

  No selection decision is produced. Never feed scores back into training.
  """
  if not _verified(reference, "final") or not _verified(selected, "final"):
    raise ValueError("Paired final comparison requires two verified final reports")
  return {
    "report_only": True,
    "selection_eligible": False,
    "protocol": reference["protocol"],
    "protocol_sha256": reference["protocol_sha256"],
    "omni": reference,
    "selected": selected,
    "scope": "Paired in-distribution draws, including three known boundary regressions",
    "interpretation": "Read survivor-only MAE with falls and coverage; no convergence claim",
  }


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

  Full validation, not smoke or final holdout, is eligible for selection.
  stage1_reference means the selected +1000 source, kept for API compatibility.
  Two successive meaningful regressions on the same metric trigger early stopping.
  These safety stopping thresholds do not require improvement.
  """
  if not _verified(reference) or not _verified(current):
    return dict(
      eligible=False, stop=True, reasons=["Unverified/incomplete/nonfinite validation"]
    )
  reasons = _degradation(reference, current, severe=True)
  if stage1_reference is not None:
    if not _verified(stage1_reference):
      return dict(
        eligible=False, stop=True, reasons=["Unverified selected +1000 source"]
      )
    reasons += [
      f"Selected +1000 retention: {r}"
      for r in _degradation(stage1_reference, current, True, include_low=True)
    ]
  if previous is not None and not _verified(previous):
    return dict(eligible=False, stop=True, reasons=["Unverified previous validation"])
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
