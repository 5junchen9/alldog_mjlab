"""合成CPU证据的评分反例；不代表任何教师具备真实动作能力。"""

import copy
import json

import pytest

from black.eval_protocol import (
  PROTOCOL,
  PROTOCOL_ID,
  QUALIFICATION_ID,
  SCHEMA,
  command,
  digest,
  roster,
)
from black.eval_score import file_sha256, score_attempt, score_trial

IDENTITY = {
  "schema": SCHEMA,
  "protocol_id": PROTOCOL_ID,
  "teacher_id": "1" * 64,
  "bundle_sha256": "2" * 64,
  "source_sha256": digest({"/fixture/source.py": "3" * 64}),
}


def make_trial(spec, ticks=None):
  ticks = spec["horizon"] if ticks is None else ticks
  rows = []
  for tick in range(ticks):
    target = command(spec, tick) if spec["kind"] == "flat" else [0.2, 0.0, 0.0]
    row = {
      "tick": tick,
      "control_tick": tick,
      "episode": 0,
      "post_episode": 0,
      "command": target,
      "velocity": list(target),
      "tilt_rad": 0.0,
      "contacts": [True, True, True, True],
      "support_proxy": True,
      "frame": tick // 5,
      "capture_tick": tick // 5 * 5,
      "depth_valid": True,
      "depth_has_hit": True,
      "action": [0.0] * 12,
      "delayed_action": [0.0] * 12,
      "processed_target": [0.0] * 12,
      "action_delay": 0,
      "terminated": False,
      "truncated": False,
      "failed": False,
    }
    if spec["kind"] == "stairs":
      stage = min(tick, 12)
      row["stair"] = {
        "ordered_treads": stage,
        "landing_contact": tick >= 7,
        "pass_once": tick >= 12,
        "invalid": False,
        "first_verified_pass_tick": 13 if tick >= 12 else -1,
        "birth": 0,
        "closed_tick": tick + 1,
        "height_cm": spec["height_cm"],
        "direction": spec["direction"],
        "angle_deg": spec["angle_deg"],
        "safe_exit": tick >= 12,
        "strict_finish": tick == 62,
        "settled_steps": max(0, tick - 12),
        "linear_speed": 0.0,
        "angular_speed": 0.0,
        "tilt_rad": 0.0,
        "contacts": [True] * 4,
      }
      if tick == 62:
        row["terminated"], row["post_episode"] = True, 1
    rows.append(row)
  end = (
    "strict_finish"
    if spec["kind"] == "stairs" and ticks == 63
    else ("horizon" if ticks == spec["horizon"] else "administrative_incomplete")
  )
  return {
    **IDENTITY,
    **{name: spec[name] for name in ("trial_id", "batch_id", "seed", "lane")},
    "episode": 0,
    "birth": 0,
    "rows": rows,
    "end_reason": end,
  }


def terminate(trial, tick, *, timeout=False):
  trial["rows"] = trial["rows"][: tick + 1]
  row = trial["rows"][-1]
  row.update(
    terminated=not timeout, failed=not timeout, truncated=timeout, post_episode=1
  )
  trial["end_reason"] = "native_timeout" if timeout else "physical_failure"


def save_attempt(directory, specs, trials, *, complete=True):
  directory.mkdir()
  manifest = {
    **IDENTITY,
    "source_files": {"/fixture/source.py": "3" * 64},
    "roster": specs,
    "action_mode": "mean",
    "qualification": QUALIFICATION_ID,
    "max_controls": None,
    "protocol": PROTOCOL,
  }
  manifest_path = directory / "MANIFEST.json"
  manifest_path.write_text(json.dumps(manifest))
  files = []
  for index, trial in enumerate(trials):
    path = directory / f"trial_{index:04d}.json"
    path.write_text(json.dumps(trial))
    files.append(
      {"trial_id": trial["trial_id"], "path": path.name, "sha256": file_sha256(path)}
    )
  result = {
    **{
      name: IDENTITY[name]
      for name in ("schema", "teacher_id", "bundle_sha256", "source_sha256")
    },
    "manifest_sha256": file_sha256(manifest_path),
    "action_mode": "mean",
    "optimizer_calls": 0,
    "teacher_unchanged": True,
    "collection_complete": complete,
    "trials": files,
  }
  result_path = directory / "RESULT.json"
  result_path.write_text(json.dumps(result))
  return manifest_path, result_path


@pytest.mark.parametrize("spec", roster("flat8", 119032)[:8])
def test_flat8_preserves_absolute_case_thresholds(spec):
  trial = make_trial(spec)
  result = score_trial(spec, trial)
  assert result["success"] and result["status"] == "pass"
  for row in trial["rows"][300:]:
    row["velocity"] = [0.0, 0.0, 0.0]
  if spec["case"] == "stop":
    for row in trial["rows"][75:150]:
      row["velocity"] = [0.0, 0.0, 0.0]
  result = score_trial(spec, trial)
  assert result["complete"] and not result["success"]


def test_failure_remains_failure_and_cannot_be_padded_by_reset_rows():
  spec = roster("flat8", 119032)[0]
  trial = make_trial(spec)
  terminate(trial, 80)
  result = score_trial(spec, trial)
  assert result["complete"] and result["physical_failure"]
  assert not result["success"] and not result["full_horizon"]
  row = copy.deepcopy(trial["rows"][-1])
  row.update(
    tick=81, control_tick=81, episode=1, post_episode=1, terminated=False, failed=False
  )
  trial["rows"].append(row)
  with pytest.raises(ValueError, match="after native terminal/reset"):
    score_trial(spec, trial)


def test_administrative_short_trace_is_incomplete():
  spec = roster("flat8", 119032)[0]
  result = score_trial(spec, make_trial(spec, 24))
  assert result["status"] == "incomplete" and not result["complete"]
  assert result["administrative_incomplete"] and not result["success"]


def test_pure_axis_start_reverse_and_full_park():
  spec = roster("axes", 119032)[0]
  trial = make_trial(spec)
  result = score_trial(spec, trial)
  assert result["success"]
  motion = result["metrics"]["motion"]
  assert [row["response_s"] for row in motion] == [0.0, 0.0]
  assert [row["steady_ticks"] for row in motion] == [200, 200]
  assert result["metrics"]["park"]["settled_at"] == 1.0
  assert result["metrics"]["park"]["ticks"] == 250
  for row in trial["rows"][100:600]:
    row["velocity"] = [0.0, 0.0, 0.0]
  failed = score_trial(spec, trial)
  assert not failed["success"]
  assert all(
    row["response_s"] is None and row["nonresponder"]
    for row in failed["metrics"]["motion"]
  )


def test_axis_park_needs_50_consecutive_supported_frames_and_all_250_rows():
  spec = roster("axes", 119032)[0]
  trial = make_trial(spec)
  for tick in range(649, 900, 50):
    trial["rows"][tick].update(
      contacts=[True, False, False, False], support_proxy=False
    )
  result = score_trial(spec, trial)
  assert result["metrics"]["park"]["settled_at"] is None
  assert not result["success"]
  trial = make_trial(spec)
  terminate(trial, 750, timeout=True)
  result = score_trial(spec, trial)
  assert result["native_timeout"] and not result["metrics"]["park"]["complete"]
  assert result["metrics"]["park"]["ticks"] == 101
  assert not result["success"]


def test_mixed_target_cannot_fill_a_pure_axis_cell():
  spec = roster("axes", 119032)[0]
  spec["target"][1] = 0.1
  with pytest.raises(ValueError, match="one pure"):
    score_trial(spec, make_trial(spec))


def test_stairs_requires_physical_passage_and_strict_finish():
  spec = roster("stairs", 119032, heights_cm=(4,), angles_deg=(0,))[0]
  trial = make_trial(spec, 63)
  assert score_trial(spec, trial)["success"]
  trial["rows"][-1]["stair"]["strict_finish"] = False
  trial["rows"][-1].update(terminated=False, truncated=True)
  trial["end_reason"] = "native_timeout"
  result = score_trial(spec, trial)
  assert result["metrics"]["physical_passage"]
  assert not result["metrics"]["strict_finish"] and not result["success"]
  trial = make_trial(spec, 63)
  for row in trial["rows"]:
    row["stair"].update(
      ordered_treads=0,
      landing_contact=False,
      pass_once=False,
      first_verified_pass_tick=-1,
    )
  result = score_trial(spec, trial)
  assert (
    not result["metrics"]["strict_finish"] and not result["metrics"]["physical_passage"]
  )
  assert not result["success"]


def test_stairs_rejects_skipped_contact_landing_and_wrong_terminal_time():
  spec = roster("stairs", 119032, heights_cm=(4,), angles_deg=(0,))[0]
  trial = make_trial(spec, 63)
  trial["rows"][3]["stair"]["ordered_treads"] = 5
  with pytest.raises(ValueError, match="skipped or lost a tread"):
    score_trial(spec, trial)
  trial = make_trial(spec, 63)
  trial["rows"][7]["stair"]["landing_contact"] = False
  with pytest.raises(ValueError, match="descent without landing"):
    score_trial(spec, trial)
  trial = make_trial(spec, 63)
  trial["rows"][-1]["stair"]["closed_tick"] += 1
  with pytest.raises(ValueError, match="terminal-before-reset"):
    score_trial(spec, trial)


def test_stair_strict_flag_requires_actual_native_terminal_and_reset():
  spec = roster("stairs", 119032, heights_cm=(4,), angles_deg=(0,))[0]
  trial = make_trial(spec, 63)
  trial["rows"][-1].update(terminated=False, post_episode=0)
  with pytest.raises(ValueError, match="strict finish requires native termination"):
    score_trial(spec, trial)


def test_stair_counter_cannot_replace_fifty_actual_supported_frames():
  spec = roster("stairs", 119032, heights_cm=(4,), angles_deg=(0,))[0]
  trial = make_trial(spec, 63)
  trial["rows"][20]["stair"]["angular_speed"] = 0.13
  with pytest.raises(ValueError, match="same-step physical support"):
    score_trial(spec, trial)
  trial = make_trial(spec, 63)
  trial["rows"][-1]["stair"]["settled_steps"] = 500
  with pytest.raises(ValueError, match="counter skipped"):
    score_trial(spec, trial)


def test_attempt_keeps_full_denominator_and_never_registers_qualification(tmp_path):
  specs = roster("axes", 119032)
  trial = make_trial(specs[0])
  paths = save_attempt(tmp_path / "attempt", specs, [trial], complete=False)
  result = score_attempt(*paths)
  assert result["counts"]["assigned"] == 24
  assert result["counts"]["started"] == result["counts"]["complete"] == 1
  assert result["counts"]["administrative_incomplete"] == 23
  assert len(result["trials"]) == 24 and result["trials"][1]["status"] == "not_started"
  assert result["qualification"] == QUALIFICATION_ID
  assert not result["automatic_promotion"] and not result["collection_complete"]


def test_attempt_reports_unassigned_pure_axis_gaps_and_invalid_trace(tmp_path):
  specs = roster("axes", 119032)[:2]
  first, second = [make_trial(spec) for spec in specs]
  second["rows"][8]["episode"] = 1
  paths = save_attempt(tmp_path / "attempt", specs, [first, second], complete=False)
  result = score_attempt(*paths)
  assert len(result["coverage"]["missing_axis_cells"]) == 22
  assert result["counts"]["invalid_evidence"] == 1
  assert result["trials"][1]["status"] == "invalid_evidence"
  assert result["counts"]["assigned"] == 2


def test_wrong_teacher_identity_is_rejected_even_with_updated_file_hash(tmp_path):
  specs = roster("flat8", 119032)[:1]
  trial = make_trial(specs[0])
  trial["teacher_id"] = "4" * 64
  paths = save_attempt(tmp_path / "attempt", specs, [trial])
  with pytest.raises(ValueError, match="trial teacher_id mismatch"):
    score_attempt(*paths)


def test_duplicate_trial_cannot_expand_denominator(tmp_path):
  specs = roster("flat8", 119032)[:1]
  trial = make_trial(specs[0])
  paths = save_attempt(tmp_path / "attempt", specs, [trial, trial])
  with pytest.raises(ValueError, match="duplicate recorded trial"):
    score_attempt(*paths)


def test_trial_hash_change_is_rejected(tmp_path):
  specs = roster("flat8", 119032)[:1]
  paths = save_attempt(tmp_path / "attempt", specs, [make_trial(specs[0])])
  with (paths[0].parent / "trial_0000.json").open("a") as stream:
    stream.write(" ")
  with pytest.raises(ValueError, match="trial SHA256 mismatch"):
    score_attempt(*paths)


def test_false_complete_summary_is_rejected(tmp_path):
  specs = roster("flat8", 119032)[:1]
  paths = save_attempt(tmp_path / "attempt", specs, [make_trial(specs[0], 24)])
  with pytest.raises(ValueError, match="collection_complete contradicts"):
    score_attempt(*paths)


def test_failed_attempt_cannot_use_an_already_published_result(tmp_path):
  specs = roster("flat8", 119032)[:1]
  paths = save_attempt(tmp_path / "attempt", specs, [make_trial(specs[0])])
  (paths[1].parent / "FAILED.json").write_text(
    json.dumps({"error": "directory fsync failed after RESULT publication"})
  )
  with pytest.raises(ValueError, match="failed attempt cannot be scored"):
    score_attempt(*paths)


@pytest.mark.parametrize(
  "field,value",
  [("optimizer_calls", 1), ("teacher_unchanged", False), ("action_mode", "sample")],
)
def test_attempt_requires_frozen_mean_policy_without_optimizer(tmp_path, field, value):
  specs = roster("flat8", 119032)[:1]
  paths = save_attempt(tmp_path / "attempt", specs, [make_trial(specs[0])])
  result = json.loads(paths[1].read_text())
  result[field] = value
  paths[1].write_text(json.dumps(result))
  with pytest.raises(ValueError):
    score_attempt(*paths)
