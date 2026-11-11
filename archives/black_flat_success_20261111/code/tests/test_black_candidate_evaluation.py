"""CPU-only protocol regression tests; no simulator or GPU initialization."""

import numpy as np
import torch

from mjlab.scripts.evaluate_black_candidates import (
  advance_command,
  candidate_scenarios,
  summarize,
  sustained_times,
)
from mjlab.scripts.evaluate_black_seed import scenarios


def test_original_cases_preserved_and_low_cases_deduplicated():
  specs = candidate_scenarios()
  assert specs[:35] == scenarios()
  assert len(specs) == 59
  assert len({tuple(s["command"]) for s in specs}) == 59
  for s in specs:
    if s["family"] == "low_mixed":
      assert np.linalg.norm(s["command"][:2]) <= 0.5
      assert abs(s["command"][2]) == 0.3


def test_ramp_vector_norm_and_yaw_caps_and_reversal_settles_before_tracking():
  current = torch.tensor([[1.0, 1.0, 0.5]])
  goal = -current.clone()
  dt = 0.02
  for _ in range(150):
    nxt = advance_command(current, goal, dt, "ramp")
    assert torch.linalg.vector_norm(nxt[:, :2] - current[:, :2]) <= dt + 1e-6
    assert abs(float(nxt[0, 2] - current[0, 2])) <= 0.5 * dt + 1e-6
    current = nxt
  assert torch.equal(current, goal)
  assert torch.equal(advance_command(-goal, goal, dt, "abrupt"), goal)


def test_sustained_time_reports_completion_and_never_success_after_fall():
  mask = np.ones((100, 2), dtype=bool)
  mask[:, 1] = False
  assert sustained_times(mask, 0.02) == [0.5, None]


def test_fallen_trials_masked_from_all_means_and_stop():
  records = {}
  for phase in ("settle", "command", "reverse", "stop"):
    valid = np.ones((300, 2), dtype=bool)
    valid[:, 1] = False
    velocity = np.zeros((300, 2, 3))
    velocity[:, 1] = 999
    records[phase] = {
      "velocity": velocity,
      "valid": valid,
      "error": np.abs(velocity),
      "stable": valid.copy(),
      "new_falls": np.array([False, phase == "command"]),
    }
  row = summarize(
    records,
    [{"name": "stand", "family": "standing", "command": [0, 0, 0]}],
    2,
    0.02,
    np.array([False, True]),
    np.array([0, 1]),
  )[0]
  assert row["trials_with_fall"] == 1
  assert row["phase_first_falls"]["command"] == 1
  assert row["command"]["mae_xyz"] == [0, 0, 0]
  assert row["stop"]["final_1s_mean_abs_velocity"] == [0, 0, 0]
  assert row["stop"]["stable_within_3s"] == 1
  assert row["reversal_time_s_each"] == [None, None]
