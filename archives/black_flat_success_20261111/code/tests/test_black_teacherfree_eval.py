import copy
from typing import Any

import pytest
import torch

from mjlab.scripts.evaluate_black_teacherfree import (
  FINAL_SEEDS,
  VALIDATION_SEEDS,
  _verified,
  aggregate_runs,
  continuation_scenarios,
  digest,
  evaluate_checkpoint,
  gate_checkpoint,
  paired_final_report,
  protocol,
  select_eligible_checkpoint,
)


def report(error=0.1, falls=0, split="validation") -> dict[str, Any]:
  p = protocol(split, False)
  runs = []
  for mode in p["modes"]:
    for seed in p["seeds"]:
      rows = []
      for spec in p["scenarios"]:
        phase = dict(
          valid_samples=100,
          total_samples=100,
          mae_xyz=[error] * 3,
          response_ratio_xyz=[1.0, None, None],
        )
        rows.append(
          dict(
            spec,
            target_command=spec["command"],
            replicates=8,
            trials_with_fall=falls,
            command=copy.deepcopy(phase),
            reverse=copy.deepcopy(phase),
            reversal_time_s_each=[1.0] * 8,
            stop=dict(settle_time_s_each=[1.0] * 8),
          )
        )
      runs.append(
        dict(
          mode=mode,
          seed=seed,
          rows=rows,
          actor_and_normalizer_unchanged=True,
          num_envs=len(rows) * 8,
          trials_with_fall=falls * len(rows),
        )
      )
  return dict(
    protocol=p,
    protocol_sha256=digest(p),
    complete=True,
    checkpoint_tensors_finite=True,
    checkpoint_unchanged=True,
    checkpoint=dict(path="checkpoint.pt", sha256="a" * 64),
    runs=runs,
    metrics=aggregate_runs(runs),
  )


def test_scenarios_are_fixed_new_draws():
  cases = continuation_scenarios()
  assert len(cases) == 28
  assert len({tuple(c["command"]) for c in cases}) == 28
  assert {c["family"] for c in cases} == {"low", "retention", "high"}
  assert cases != continuation_scenarios("final")
  assert not set(VALIDATION_SEEDS) & set(FINAL_SEEDS)
  assert not (set(VALIDATION_SEEDS) | set(FINAL_SEEDS)) & {81101, 81102}


def test_unchanged_baseline_safe_without_low_speed_improvement():
  ref = report()
  assert gate_checkpoint(ref, copy.deepcopy(ref))["eligible"]
  assert select_eligible_checkpoint(ref, [copy.deepcopy(ref)]) == ref["checkpoint"]


@pytest.mark.parametrize("error,falls", [(0.136, 0), (0.1, 1)])
def test_severe_regression_stops(error, falls):
  assert gate_checkpoint(report(), report(error, falls))["stop"]


def test_repeated_deterioration_not_small_noise():
  ref = report()
  moderate = report(0.12)
  assert gate_checkpoint(ref, moderate)["eligible"]
  assert gate_checkpoint(ref, moderate, moderate)["stop"]
  assert gate_checkpoint(ref, report(0.105), report(0.105))["eligible"]
  assert select_eligible_checkpoint(ref, [moderate, moderate]) == moderate["checkpoint"]


@pytest.mark.parametrize(
  "fault", ["smoke", "final", "nonfinite", "missing", "actor", "metrics"]
)
def test_invalid_report_not_eligible(fault):
  ref, current = report(), report()
  if fault == "smoke":
    current["protocol"]["smoke"] = True
  elif fault == "final":
    current["protocol"]["split"] = "final"
  elif fault == "nonfinite":
    current["metrics"]["all"]["low"]["tracking_mae"] = float("nan")
  elif fault == "missing":
    current["runs"].pop()
  elif fault == "actor":
    current["runs"][0]["actor_and_normalizer_unchanged"] = False
  else:
    current["metrics"]["all"]["low"]["tracking_mae"] = 0.0
  assert not gate_checkpoint(ref, current)["eligible"]


def test_aggregate_counts_and_weighting():
  current = report(0.2, 1)
  metric = current["metrics"]["all"]["all"]
  assert metric["trials"] == 28 * 8 * 4
  assert metric["fall_fraction"] == 0.125
  assert metric["tracking_mae"] == pytest.approx(0.2)
  assert metric["reversal_success_fraction"] == 1.0


def test_bounds_and_boundary_regression_labels():
  for split in ("validation", "final"):
    for case in continuation_scenarios(split):
      x, y, yaw = case["command"]
      assert abs(x) <= 1 and abs(y) <= 1 and abs(yaw) <= 0.5
      if abs(x) == 1:
        assert case["previously_seen_boundary_regression"]


def test_stage2_preserves_low_stage1_skill():
  baseline, stage1, current = report(0.2), report(0.1), report(0.1)
  for run in current["runs"]:
    for row in run["rows"]:
      if row["family"] == "low":
        for phase in ("command", "reverse"):
          row[phase]["mae_xyz"] = [0.15] * 3
  current["metrics"] = aggregate_runs(current["runs"])
  assert gate_checkpoint(baseline, current)["eligible"]
  assert gate_checkpoint(baseline, current, stage1_reference=stage1)["stop"]


def test_all_falls_no_survivors_rejected_without_exception():
  current = report()
  for run in current["runs"]:
    for row in run["rows"]:
      if row["family"] == "high":
        row["trials_with_fall"] = 8
        for phase in ("command", "reverse"):
          row[phase].update(
            mae_xyz=None, valid_samples=0, response_ratio_xyz=[None] * 3
          )
  current["metrics"] = aggregate_runs(current["runs"])
  assert gate_checkpoint(report(), current)["stop"]


def test_balanced_retention_and_known_boundaries():
  cases = continuation_scenarios()
  pure = [
    r["command"]
    for r in cases
    if r["family"] == "retention" and sum(v != 0 for v in r["command"]) == 1
  ]
  for axis in range(3):
    assert any(c[axis] < 0 for c in pure)
    assert any(c[axis] > 0 for c in pure)
  assert sum(c.get("previously_seen_boundary_regression", False) for c in cases) == 3
  assert VALIDATION_SEEDS == (941101, 941102)
  assert FINAL_SEEDS == (951101, 951102)


@pytest.mark.parametrize(
  "fault",
  ["row", "target", "replicates", "reversal", "stop", "sha", "samples", "duplicate"],
)
def test_full_coverage_required(fault):
  r = report()
  row = r["runs"][0]["rows"][0]
  if fault == "row":
    r["runs"][0]["rows"].pop()
  elif fault == "target":
    row["target_command"] = [9, 9, 9]
  elif fault == "replicates":
    row["replicates"] = 1
  elif fault == "reversal":
    row["reversal_time_s_each"].pop()
  elif fault == "stop":
    row["stop"]["settle_time_s_each"].pop()
  elif fault == "sha":
    r["checkpoint"]["sha256"] = "invalid"
  elif fault == "samples":
    row["reverse"]["valid_samples"] = -1
  else:
    r["runs"][1] = copy.deepcopy(r["runs"][0])
  r["metrics"] = aggregate_runs(r["runs"])
  assert not _verified(r)
  assert gate_checkpoint(report(), r)["stop"]


def test_final_paired_report_is_never_gate_eligible():
  omni, selected = report(split="final"), report(split="final")
  paired = paired_final_report(omni, selected)
  assert paired["report_only"] and not paired["selection_eligible"]
  assert _verified(selected, "final")
  assert not _verified(selected)
  assert gate_checkpoint(omni, selected)["stop"]
  with pytest.raises(ValueError):
    paired_final_report(report(), selected)
  assert not _verified(report(), smoke=True)


def test_invalid_previous_cannot_reset_repeated_regression():
  previous = report()
  previous["complete"] = False
  assert gate_checkpoint(report(), report(), previous)["stop"]


@pytest.mark.parametrize("family", ["low", "retention", "high"])
def test_repeated_source_forgetting_stops_all_families(family):
  omni, source, current = report(0.2), report(0.1), report(0.1)
  for run in current["runs"]:
    for row in run["rows"]:
      if row["family"] == family:
        for phase in ("command", "reverse"):
          row[phase]["mae_xyz"] = [0.12] * 3
  current["metrics"] = aggregate_runs(current["runs"])
  assert gate_checkpoint(omni, current, stage1_reference=source)["eligible"]
  assert gate_checkpoint(omni, current, current, source)["stop"]


def test_evaluate_keeps_target_and_verifies_immutable_checkpoint(tmp_path, monkeypatch):
  from mjlab.scripts import evaluate_black_teacherfree as module

  checkpoint = tmp_path / "model.pt"
  torch.save(
    {
      "actor_state_dict": {"w": torch.zeros(1)},
      "critic_state_dict": {"w": torch.ones(1)},
    },
    checkpoint,
  )
  fixture = report()
  calls = []

  def fake_evaluate(args, seed, specs):
    calls.append((args.mode, seed, args.replicates))
    run = next(
      copy.deepcopy(r)
      for r in fixture["runs"]
      if r["mode"] == args.mode and r["seed"] == seed
    )
    run.pop("mode")
    for row in run["rows"]:
      row.pop("target_command")
    return run

  monkeypatch.setattr(module, "evaluate", fake_evaluate)
  output = tmp_path / "report.json"
  result = evaluate_checkpoint(checkpoint, output)
  assert _verified(result)
  assert len(calls) == 4 and all(c[2] == 8 for c in calls)
  assert isinstance(result["runs"][0]["rows"][0]["command"], dict)
  assert result["runs"][0]["rows"][0]["target_command"] == [-0.07, 0, 0]
  with pytest.raises(FileExistsError):
    evaluate_checkpoint(checkpoint, output)

  def mutation(args, seed, specs):
    checkpoint.write_bytes(b"changed checkpoint")
    return fake_evaluate(args, seed, specs)

  monkeypatch.setattr(module, "evaluate", mutation)
  with pytest.raises(RuntimeError, match="Checkpoint changed"):
    evaluate_checkpoint(checkpoint, tmp_path / "mutated.json")
  assert not (tmp_path / "mutated.json").exists()


@pytest.mark.parametrize("key", ["actor_state_dict", "critic_state_dict"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_checkpoint_rejected_before_rollout(tmp_path, monkeypatch, key, bad):
  from mjlab.scripts import evaluate_black_teacherfree as module

  payload = {
    "actor_state_dict": {"w": torch.zeros(1)},
    "critic_state_dict": {"w": torch.zeros(1)},
  }
  payload[key]["w"][0] = bad
  checkpoint = tmp_path / "nonfinite.pt"
  torch.save(payload, checkpoint)

  def forbidden(*args, **kwargs):
    pytest.fail("Rollout must not start for a nonfinite checkpoint")

  monkeypatch.setattr(module, "evaluate", forbidden)
  with pytest.raises(FloatingPointError, match=key):
    evaluate_checkpoint(checkpoint, tmp_path / "report.json")
  assert not (tmp_path / "report.json").exists()
