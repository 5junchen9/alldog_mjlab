"""用合成完整原始面板测试门；通过fixture不授予任何真实教师资格。"""

import json

import pytest
from eval_protocol import EVALUATION_SEEDS, QUALIFICATION_ID, roster
from eval_score import file_sha256
from qualification import require_qualification
from test_eval_score import IDENTITY, make_trial, save_attempt


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
  root = tmp_path_factory.mktemp("qualification_raw")
  entries = []
  for suite in ("flat8", "axes", "stairs"):
    for seed in EVALUATION_SEEDS:
      specs = roster(suite, seed)
      trials = [make_trial(spec, 63 if suite == "stairs" else None) for spec in specs]
      directory = root / f"{suite}_{seed}"
      manifest_path, result_path = save_attempt(directory, specs, trials)
      manifest = json.loads(manifest_path.read_text())
      manifest.update(suite=suite, base_seed=seed)
      manifest_path.write_text(json.dumps(manifest))
      result = json.loads(result_path.read_text())
      result["manifest_sha256"] = file_sha256(manifest_path)
      result_path.write_text(json.dumps(result))
      entries.append(
        dict(
          path=str(directory),
          manifest_sha256=file_sha256(manifest_path),
          result_sha256=file_sha256(result_path),
        )
      )
  return root, entries


def check(root, entries, stage, **kwargs) -> dict:
  path = root / "INDEX.json"
  path.write_text(
    json.dumps(dict(schema=QUALIFICATION_ID, attempts=entries, passed=True))
  )
  return require_qualification(
    path,
    kwargs.get("teacher", IDENTITY["teacher_id"]),
    stage,
    source_sha256=kwargs.get("source", IDENTITY["source_sha256"]),
  )


def test_all_three_fixed_seed_full_panels_recompute_from_raw(panel):
  root, entries = panel
  report: dict = {}
  for stage, count in (("flat_axes", 3), ("stairs", 6), ("student", 9)):
    report = check(root, entries[:count], stage)
    assert report["passed"] and report["recomputed"]
  assert sum(row["assigned"] for row in report["attempts"]) == 3 * (32 + 24 + 54)


def test_handfilled_success_partial_panel_duplicate_and_wrong_identity_rejected(panel):
  root, entries = panel
  for selected in ([], entries[:1], entries[:2], entries[:2] + entries[:1]):
    with pytest.raises(ValueError):
      check(root, selected, "flat_axes")
  with pytest.raises(ValueError, match="another teacher"):
    check(root, entries[:3], "flat_axes", teacher="other")
  with pytest.raises(ValueError, match="current source"):
    check(root, entries[:3], "flat_axes", source="other")


def test_score_summary_cannot_hide_failed_or_changed_raw_attempt(panel):
  root, entries = panel
  from pathlib import Path

  directory = Path(entries[0]["path"])
  failed = directory / "FAILED.json"
  failed.write_text('{"error":"capture failed"}')
  (directory / "SCORE.json").write_text('{"passed":true}')
  try:
    with pytest.raises(ValueError, match="failed attempt"):
      check(root, entries[:3], "flat_axes")
  finally:
    failed.unlink()
  trial = directory / "trial_0000.json"
  original = trial.read_bytes()
  try:
    trial.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="trial SHA256"):
      check(root, entries[:3], "flat_axes")
  finally:
    trial.write_bytes(original)


def test_rehashed_failed_motion_still_blocks_admission(panel):
  from pathlib import Path

  root, entries = panel
  directory = Path(entries[0]["path"])
  trial_path, result_path = directory / "trial_0000.json", directory / "RESULT.json"
  old_trial, old_result = trial_path.read_bytes(), result_path.read_bytes()
  try:
    trial = json.loads(old_trial)
    for row in trial["rows"]:
      row["velocity"] = [0, 0, 0]
    trial_path.write_text(json.dumps(trial))
    result = json.loads(old_result)
    result["trials"][0]["sha256"] = file_sha256(trial_path)
    result_path.write_text(json.dumps(result))
    selected = [dict(row) for row in entries[:3]]
    selected[0]["result_sha256"] = file_sha256(result_path)
    with pytest.raises(ValueError, match="failed or invalid trial"):
      check(root, selected, "flat_axes")
  finally:
    trial_path.write_bytes(old_trial)
    result_path.write_bytes(old_result)
