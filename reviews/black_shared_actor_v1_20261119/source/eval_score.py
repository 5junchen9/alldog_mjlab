"""仅用保存的证据计算绝对动作指标；不登记教师总体资格。"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from eval_protocol import (
  FLAT8,
  HEIGHTS_CM,
  PROTOCOL,
  PROTOCOL_ID,
  SCHEMA,
  command,
  digest,
)

FLAT_CASES = tuple(FLAT8)
IDENTITY_FIELDS = (
  "schema",
  "protocol_id",
  "teacher_id",
  "bundle_sha256",
  "source_sha256",
)


def require(condition, message):
  if not condition:
    raise ValueError(message)


def file_sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
  return sum(values) / len(values) if values else None


def finite_vector(value, size):
  return (
    isinstance(value, list)
    and len(value) == size
    and all(type(x) in (int, float) and math.isfinite(x) for x in value)
  )


def validate_spec(spec):
  program = spec["program"]
  require(program in ("flat8", "axis", "stairs"), "unknown program")
  require(
    spec["kind"] == ("stairs" if program == "stairs" else "flat"),
    "kind/program mismatch",
  )
  require(
    spec["horizon"] == {"flat8": 400, "axis": 900, "stairs": 6000}[program],
    "horizon mismatch",
  )
  for name in ("batch_id", "seed", "lane"):
    require(type(spec[name]) is int and spec[name] >= 0, f"invalid {name}")
  require(spec["lane"] < 32, "lane outside fixed batch")
  if program == "flat8":
    require(spec["case"] in FLAT_CASES, "unknown flat8 case")
    require(spec["target"] == FLAT8[spec["case"]], "flat8 target mismatch")
  elif program == "axis":
    target = spec["target"]
    require(finite_vector(target, 3), "invalid axis target")
    active = [x for x in target if x != 0]
    require(
      len(active) == 1
      and any(math.isclose(abs(active[0]), x, abs_tol=1e-7) for x in (0.1, 0.25)),
      "axis program requires one pure signed 0.10/0.25 target",
    )
    require(spec["orientation"] in (0.0, math.pi), "axis orientation must be 0 or pi")
  else:
    require(spec["direction"] in (-1, 1), "stairs direction must be signed")
    require(spec["height_cm"] in HEIGHTS_CM, "unknown stairs height")
    require(spec["angle_deg"] in (-15, 0, 15), "unregistered stairs angle")


def validate_rows(spec, trial):
  require(
    trial["schema"] == SCHEMA and trial["protocol_id"] == PROTOCOL_ID,
    "trial schema/protocol mismatch",
  )
  for name in ("trial_id", "batch_id", "seed", "lane"):
    require(trial[name] == spec[name], f"trial {name} mismatch")
  for name in ("episode", "birth"):
    require(type(trial[name]) is int and trial[name] >= 0, f"invalid trial {name}")
  rows = trial["rows"]
  require(
    isinstance(rows, list) and 0 < len(rows) <= spec["horizon"],
    "invalid observed row count",
  )
  first_control = rows[0]["control_tick"]
  require(
    type(first_control) is int and first_control >= trial["birth"],
    "invalid first control tick",
  )
  ended = False
  previous_stage, previous_landing, previous_pass = 0, False, False
  first_pass_tick = -1
  for tick, row in enumerate(rows):
    require(not ended, "rows after native terminal/reset")
    require(
      type(row["tick"]) is int and row["tick"] == tick, "noncontiguous trial ticks"
    )
    require(
      row["control_tick"] == first_control + tick, "noncontiguous native control clock"
    )
    require(
      row["episode"] == trial["episode"], "reset episode cannot supplement a trial"
    )
    for name in ("terminated", "truncated", "failed", "support_proxy", "depth_valid"):
      require(type(row[name]) is bool, f"invalid {name} flag")
    done = row["terminated"] or row["truncated"]
    require(
      row["post_episode"] == row["episode"] + int(done), "post-reset episode mismatch"
    )
    require(finite_vector(row["velocity"], 3), "invalid pre-action IMU velocity")
    require(finite_vector(row["command"], 3), "invalid command")
    require(finite_vector(row["action"], 12), "invalid mean action")
    for name in ("delayed_action", "processed_target"):
      require(finite_vector(row[name], 12), f"invalid {name}")
    require(
      type(row["action_delay"]) is int and 0 <= row["action_delay"] <= 2,
      "invalid native action delay",
    )
    require(
      type(row["tilt_rad"]) in (int, float)
      and math.isfinite(row["tilt_rad"])
      and 0 <= row["tilt_rad"] <= math.pi + 1e-5,
      "invalid tilt",
    )
    contacts = row["contacts"]
    require(
      isinstance(contacts, list)
      and len(contacts) == 4
      and all(type(x) is bool for x in contacts),
      "invalid native foot contacts",
    )
    require(
      row["support_proxy"] == (sum(contacts) >= 2), "support proxy/contact mismatch"
    )
    require(type(row["frame"]) is int and row["frame"] >= -1, "invalid camera frame")
    require(
      type(row["capture_tick"]) is int and row["capture_tick"] <= row["control_tick"],
      "future camera frame",
    )
    if row["depth_valid"]:
      require(
        row["frame"] >= 0 and row["capture_tick"] >= 0,
        "valid depth lacks frame identity",
      )
    if spec["kind"] == "flat":
      expected = command(spec, tick)
      require(
        all(
          math.isclose(x, y, rel_tol=0, abs_tol=1e-6)
          for x, y in zip(row["command"], expected, strict=True)
        ),
        "command does not match registered tape",
      )
    else:
      proof = row["stair"]
      for name in ("height_cm", "direction", "angle_deg"):
        require(proof[name] == spec[name], f"stairs {name} differs from assigned route")
      require(proof["birth"] == trial["birth"], "stairs birth mismatch")
      require(
        proof["closed_tick"] == row["control_tick"] + 1,
        "stairs proof not from terminal-before-reset step",
      )
      require(
        type(proof["ordered_treads"]) is int and 0 <= proof["ordered_treads"] <= 12,
        "invalid ordered tread count",
      )
      require(
        type(proof["settled_steps"]) is int and proof["settled_steps"] >= 0,
        "invalid settled steps",
      )
      require(
        type(proof["first_verified_pass_tick"]) is int, "invalid first passage tick"
      )
      for name in (
        "landing_contact",
        "pass_once",
        "invalid",
        "safe_exit",
        "strict_finish",
      ):
        require(type(proof[name]) is bool, f"invalid passage {name}")
      require(
        not proof["strict_finish"] or row["terminated"],
        "strict finish requires native termination and reset",
      )
      stage, landing, passed = (
        proof["ordered_treads"],
        proof["landing_contact"],
        proof["pass_once"],
      )
      require(
        previous_stage <= stage <= previous_stage + 1,
        "stairs sequence skipped or lost a tread",
      )
      require(not previous_landing or landing, "landing evidence disappeared")
      require(
        not landing or previous_landing or previous_stage == 6,
        "landing before ordered ascent",
      )
      require(stage <= 6 or landing, "descent without landing contact")
      require(not previous_pass or passed, "passage evidence disappeared")
      if passed:
        require(
          stage == 12 and landing, "passage lacks twelve ordered treads and landing"
        )
        if not previous_pass:
          first_pass_tick = proof["closed_tick"]
          require(
            proof["safe_exit"] and not proof["invalid"], "first passage lacks safe exit"
          )
        require(
          proof["birth"] < proof["first_verified_pass_tick"] == first_pass_tick,
          "first passage timestamp mismatch",
        )
      else:
        require(
          proof["first_verified_pass_tick"] == -1, "unverified passage has a timestamp"
        )
      previous_stage, previous_landing, previous_pass = stage, landing, passed
    ended = (
      done
      or row["failed"]
      or (spec["kind"] == "stairs" and row["stair"]["strict_finish"])
    )
  final = rows[-1]
  strict = spec["kind"] == "stairs" and final["stair"]["strict_finish"]
  if final["failed"] or (final["terminated"] and not strict):
    expected_end = "physical_failure"
  elif strict:
    expected_end = "strict_finish"
  elif final["truncated"]:
    expected_end = "native_timeout"
  elif len(rows) == spec["horizon"]:
    expected_end = "horizon"
  else:
    expected_end = "administrative_incomplete"
  require(
    trial["end_reason"] == expected_end,
    "end reason does not match actual terminal evidence",
  )


def flat8_metrics(spec, rows, eligible):
  velocity = [row["velocity"] for row in rows]
  late = velocity[300:400]
  entry = mean([v[0] for v in velocity[75:150]])
  criteria = {}
  if spec["case"] in ("reversal", "stop"):
    criteria["forward_entry"] = eligible and entry is not None and entry >= 0.15
    if spec["case"] == "reversal":
      reverse = mean([-v[0] for v in late])
      criteria["reverse"] = eligible and reverse is not None and reverse >= 0.15
    else:
      xy, yaw = (
        mean([math.hypot(*v[:2]) for v in late]),
        mean([abs(v[2]) for v in late]),
      )
      criteria["stop"] = (
        eligible and xy is not None and yaw is not None and xy <= 0.10 and yaw <= 0.15
      )
  else:
    target = command(spec, 0)
    axis = max(range(3), key=lambda i: abs(target[i]))
    directed = mean([v[axis] * math.copysign(1, target[axis]) for v in late])
    criteria["signed_motion"] = (
      eligible and directed is not None and directed >= abs(target[axis]) * 0.5
    )
  return {
    "criteria": criteria,
    "late_velocity_mean": [mean([v[i] for v in late]) for i in range(3)],
    "forward_entry_vx": entry,
    "passed": eligible and all(criteria.values()),
  }


def response_latency(values, floor):
  return next(
    (0.02 * i for i in range(len(values) - 9) if min(values[i : i + 10]) >= floor), None
  )


def axis_metrics(spec, rows, eligible):
  motion = []
  for phase, lo, hi in (("start", 100, 350), ("reverse", 350, 600)):
    target = command(spec, lo)
    axis = next(i for i, value in enumerate(target) if value)
    magnitude = abs(target[axis])
    values = [
      math.copysign(1, target[axis]) * row["velocity"][axis] for row in rows[lo:hi]
    ]
    steady = values[50:]
    average = mean(steady)
    full = len(values) == 250 and not any(
      row["terminated"] or row["truncated"] or row["failed"] for row in rows[lo:hi]
    )
    latency = response_latency(values, magnitude * 0.5)
    motion.append(
      {
        "phase": phase,
        "axis": axis,
        "target": target[axis],
        "ticks": len(values),
        "steady_ticks": len(steady),
        "complete": full,
        "response_s": latency,
        "nonresponder": latency is None,
        "steady_directed": average,
        "mae": mean([abs(v - magnitude) for v in steady]),
        "overshoot_count": sum(v > magnitude + 0.05 for v in steady),
        "overshoot_denominator": len(steady),
        "overshoot_qualification": "not_registered",
        "passed": full and average is not None and average >= magnitude * 0.5,
      }
    )
  park = rows[650:900]
  xy = mean([math.hypot(*row["velocity"][:2]) for row in park])
  yaw = mean([abs(row["velocity"][2]) for row in park])
  streak, settled_at = 0, None
  for tick, row in enumerate(rows[600:900], start=600):
    valid = (
      not (row["terminated"] or row["truncated"] or row["failed"])
      and math.hypot(*row["velocity"][:2]) <= 0.08
      and abs(row["velocity"][2]) <= 0.12
      and row["tilt_rad"] <= 0.35
      and row["support_proxy"]
    )
    streak = streak + 1 if valid else 0
    if streak == 26 and settled_at is None:
      settled_at = (tick - 600) * 0.02
  park_passed = (
    eligible
    and len(park) == 250
    and xy is not None
    and yaw is not None
    and xy <= 0.10
    and yaw <= 0.15
    and settled_at is not None
  )
  return {
    "motion": motion,
    "park": {
      "ticks": len(park),
      "complete": eligible and len(park) == 250,
      "planar": xy,
      "yaw": yaw,
      "settled_at": settled_at,
      "passed": park_passed,
    },
    "passed": eligible and park_passed and all(item["passed"] for item in motion),
  }


def stairs_metrics(trial):
  proof = trial["rows"][-1]["stair"]
  passage = (
    proof["ordered_treads"] == 12
    and proof["landing_contact"]
    and proof["pass_once"]
    and not proof["invalid"]
    and proof["safe_exit"]
    and proof["birth"] < proof["first_verified_pass_tick"] <= proof["closed_tick"]
  )
  strict = proof["strict_finish"] and proof["settled_steps"] >= 6
  return {
    "physical_passage": passage,
    "strict_finish": strict,
    "proof": proof,
    "passed": passage and strict and trial["end_reason"] == "strict_finish",
    "course_success_is_qualification": False,
  }


def score_trial(spec, trial):
  """一条完整episode证据；非法身份/时钟拒绝，行政短程保留incomplete。"""
  validate_spec(spec)
  validate_rows(spec, trial)
  rows = trial["rows"]
  end = trial["end_reason"]
  eligible = end == "horizon" and len(rows) == spec["horizon"]
  if spec["program"] == "flat8":
    metrics = flat8_metrics(spec, rows, eligible)
  elif spec["program"] == "axis":
    metrics = axis_metrics(spec, rows, eligible)
  else:
    metrics = stairs_metrics(trial)
  complete = end != "administrative_incomplete"
  success = complete and metrics["passed"]
  return {
    "trial_id": spec["trial_id"],
    "program": spec["program"],
    "status": "pass" if success else "fail" if complete else "incomplete",
    "started": True,
    "complete": complete,
    "success": success,
    "physical_failure": end == "physical_failure",
    "native_timeout": end == "native_timeout",
    "administrative_incomplete": not complete,
    "invalid_evidence": False,
    "end_reason": end,
    "observed_ticks": len(rows),
    "full_horizon": len(rows) == spec["horizon"],
    "episode_identity": [
      trial[name] for name in ("batch_id", "seed", "lane", "episode", "birth")
    ],
    "invalid_depth_frames": sum(not row["depth_valid"] for row in rows),
    "metrics": metrics,
  }


def absent_trial(spec, error=None):
  return {
    "trial_id": spec["trial_id"],
    "program": spec["program"],
    "status": "invalid_evidence" if error else "not_started",
    "started": error is not None,
    "complete": False,
    "success": False,
    "physical_failure": False,
    "native_timeout": False,
    "administrative_incomplete": error is None,
    "invalid_evidence": error is not None,
    "error": error,
  }


def coverage(roster, scored):
  """逐格留分母；组合运动不会补纯轴格，也不据此创造总通过率门。"""
  groups = defaultdict(list)
  for spec, result in zip(roster, scored, strict=True):
    if spec["program"] == "flat8":
      key = f"flat8/{spec['case']}"
    elif spec["program"] == "axis":
      axis = next(i for i, value in enumerate(spec["target"]) if value)
      key = f"axis/{axis}/{spec['target'][axis]:+.2f}/{float(spec['orientation'])}"
    else:
      key = f"stairs/{spec['height_cm']}/{spec['direction']}/{spec['angle_deg']}"
    groups[key].append(result)
  cells = {
    key: {
      "assigned": len(items),
      "complete": sum(x["complete"] for x in items),
      "success": sum(x["success"] for x in items),
      "qualification": "not_registered",
    }
    for key, items in sorted(groups.items())
  }
  required_axis = [
    f"axis/{axis}/{sign * magnitude:+.2f}/{orientation}"
    for axis in range(3)
    for magnitude in (0.1, 0.25)
    for sign in (-1, 1)
    for orientation in (0.0, math.pi)
  ]
  return {
    "cells": cells,
    "missing_flat8_cells": [
      f"flat8/{case}" for case in FLAT_CASES if f"flat8/{case}" not in cells
    ],
    "missing_axis_cells": [key for key in required_axis if key not in cells],
    "stairs_qualification_scope": "not_registered",
  }


def score_attempt(manifest_path, result_path):
  """校验文件链与固定分母；只读历史记录，不导入Torch或创建环境。"""
  manifest_path, result_path = Path(manifest_path), Path(result_path)
  require(
    not (result_path.parent / "FAILED.json").exists(),
    "failed attempt cannot be scored as a completed result",
  )
  manifest = json.loads(manifest_path.read_text())
  result = json.loads(result_path.read_text())
  require(
    manifest["schema"] == SCHEMA and manifest["protocol_id"] == PROTOCOL_ID,
    "manifest schema/protocol mismatch",
  )
  require(
    manifest["qualification"] == "not_registered", "unregistered qualification required"
  )
  require(
    digest(manifest["protocol"]) == digest(PROTOCOL),
    "absolute threshold protocol mismatch",
  )
  require(
    result["manifest_sha256"] == file_sha256(manifest_path), "manifest SHA256 mismatch"
  )
  require(
    manifest["source_sha256"] == digest(manifest["source_files"]),
    "source file manifest digest mismatch",
  )
  for name in IDENTITY_FIELDS:
    if name == "protocol_id":  # RESULT通过manifest SHA绑定协议。
      continue
    require(result[name] == manifest[name], f"result {name} mismatch")
  require(
    manifest["action_mode"] == result["action_mode"] == "mean",
    "only policy mean evaluation is supported",
  )
  require(
    type(result["optimizer_calls"]) is int and result["optimizer_calls"] == 0,
    "evaluation performed optimizer updates",
  )
  require(result["teacher_unchanged"] is True, "teacher changed during evaluation")
  require(
    type(result["collection_complete"]) is bool, "invalid collection completion flag"
  )
  roster = manifest["roster"]
  require(isinstance(roster, list) and roster, "nonempty assigned roster required")
  ids, slots = set(), set()
  for spec in roster:
    validate_spec(spec)
    slot = tuple(spec[name] for name in ("batch_id", "seed", "lane"))
    require(
      spec["trial_id"] not in ids and slot not in slots,
      "duplicate assigned trial or batch lane",
    )
    ids.add(spec["trial_id"])
    slots.add(slot)
  records, paths = {}, set()
  for record in result["trials"]:
    trial_id, name = record["trial_id"], record["path"]
    require(
      trial_id in ids and trial_id not in records, "unknown or duplicate recorded trial"
    )
    require(
      isinstance(name, str)
      and Path(name).name == name
      and Path(name).suffix == ".json"
      and name not in paths,
      "invalid or duplicate trial file",
    )
    path = result_path.parent / name
    require(file_sha256(path) == record["sha256"], "trial SHA256 mismatch")
    trial = json.loads(path.read_text())
    for field in IDENTITY_FIELDS:
      require(trial[field] == manifest[field], f"trial {field} mismatch")
    require(trial["trial_id"] == trial_id, "trial file identity mismatch")
    records[trial_id] = trial
    paths.add(name)
  scored, episodes = [], set()
  for spec in roster:
    trial = records.get(spec["trial_id"])
    if trial is None:
      scored.append(absent_trial(spec))
      continue
    try:
      item = score_trial(spec, trial)
    except (ValueError, KeyError, TypeError) as error:
      scored.append(absent_trial(spec, str(error)))
      continue
    episode = tuple(item["episode_identity"])
    require(episode not in episodes, "duplicate independent episode")
    episodes.add(episode)
    scored.append(item)
  counts = {"assigned": len(roster), "independent_episodes": len(episodes)}
  for key in (
    "started",
    "complete",
    "success",
    "physical_failure",
    "native_timeout",
    "administrative_incomplete",
    "invalid_evidence",
  ):
    counts[key] = sum(item[key] for item in scored)
  complete = counts["complete"] == counts["assigned"] and not counts["invalid_evidence"]
  require(
    not result["collection_complete"] or complete,
    "collection_complete contradicts assigned trial evidence",
  )
  return {
    "schema": SCHEMA,
    "protocol_id": PROTOCOL_ID,
    "manifest_sha256": file_sha256(manifest_path),
    "result_sha256": file_sha256(result_path),
    "teacher_id": manifest["teacher_id"],
    "bundle_sha256": manifest["bundle_sha256"],
    "source_sha256": manifest["source_sha256"],
    "qualification": "not_registered",
    "collection_complete": complete,
    "counts": counts,
    "trials": scored,
    "coverage": coverage(roster, scored),
    "automatic_promotion": False,
    "status_counts": dict(Counter(item["status"] for item in scored)),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--result", type=Path, required=True)
  parser.add_argument("--out", type=Path)
  args = parser.parse_args()
  report = score_attempt(args.manifest, args.result)
  text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
  if args.out is None:
    print(text, end="")
  else:
    with args.out.open("x") as stream:
      stream.write(text)


if __name__ == "__main__":
  main()
