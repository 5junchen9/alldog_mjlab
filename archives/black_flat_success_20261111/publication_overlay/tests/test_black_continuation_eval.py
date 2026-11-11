import copy
from typing import Any

import pytest

from mjlab.scripts.evaluate_black_continuation import (
  FINAL_SEEDS,
  VALIDATION_SEEDS,
  aggregate_runs,
  continuation_scenarios,
  digest,
  gate_checkpoint,
  protocol,
  select_eligible_checkpoint,
)


def report(error=0.1, falls=0) -> dict[str, Any]:
  p = protocol("validation", False)
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
            replicates=8,
            trials_with_fall=falls,
            command=copy.deepcopy(phase),
            reverse=copy.deepcopy(phase),
            reversal_time_s_each=[1.0] * 8,
            stop=dict(settle_time_s_each=[1.0] * 8),
          )
        )
      runs.append(
        dict(mode=mode, seed=seed, rows=rows, actor_and_normalizer_unchanged=True)
      )
  return dict(
    protocol=p,
    protocol_sha256=digest(p),
    complete=True,
    checkpoint_unchanged=True,
    checkpoint=dict(path="checkpoint.pt", sha256="a"),
    runs=runs,
    metrics=aggregate_runs(runs),
  )


def test_scenarios_are_fixed_new_draws():
  cases = continuation_scenarios()
  assert len(cases) == 24
  assert len({tuple(c["command"]) for c in cases}) == 24
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
  assert metric["trials"] == 24 * 8 * 4
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
