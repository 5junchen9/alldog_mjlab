"""只从完整原始轨迹计算能力；不接受手填passed作为资格。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wolf.checkpoint import CONTRACT, source_identity
from wolf.curriculum import SPECIALTIES
from wolf.eval_protocol import PROTOCOL, SEEDS, panel, requirements


def digest(value):
  return hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()


def geometry_contract(spec):
  if spec["course"] not in SPECIALTIES:
    return None
  import re

  import mujoco
  from support.black.obstacle_geometry import historical_case
  from support.black.sparse_support import support_plane

  model, data, case = historical_case(spec["case"])
  tx = 0.75 - case["xmin"]
  ty = -case["y"]
  treads = []
  surfaces = []
  for gid in range(model.ngeom):
    name = model.geom(gid).name
    if "stairs_" in name and (
      "_up_" in name or "_down_" in name or name.endswith("_landing")
    ):
      if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("楼梯证据只支持原有箱顶")
      x, y, z = data.geom_xpos[gid]
      sx, sy, sz = model.geom_size[gid]
      treads.append(
        [
          float(x - sx + tx),
          float(x + sx + tx),
          float(y - sy + ty),
          float(y + sy + ty),
          float(z + sz),
        ]
      )
    if spec["course"] in ("broken_bridge", "plum_stakes") and re.match(
      r"^(?:trench_\d+cm|plum(?:_t\d+)?)_", name
    ):
      bounds, plane = support_plane(model, data, gid)
      nx, ny, nz, d_world = [float(v) for v in plane]
      d_local = d_world - nx * tx - ny * ty
      surfaces.append(
        [
          float(bounds[0] + tx),
          float(bounds[1] + tx),
          float(bounds[2] + ty),
          float(bounds[3] + ty),
          nx,
          ny,
          nz,
          float(d_local),
        ]
      )
  treads.sort(key=lambda b: b[0], reverse=spec["direction"] < 0)
  core = case.get("core")
  return {
    "length": case["xmax"] - case["xmin"] + 1.5,
    "width": case["half_width"],
    "core": [float(core[0] + tx), float(core[1] + tx)] if core is not None else None,
    "surfaces": surfaces,
    "treads": treads,
  }


def in_surface(points, bounds):
  if len(bounds) == 8:
    x0, x1, y0, y1, nx, ny, nz, d = bounds
    distance = np.abs(points[:, 0] * nx + points[:, 1] * ny + points[:, 2] * nz + d)
    return (
      (points[:, 0] >= x0 - 0.02)
      & (points[:, 0] <= x1 + 0.02)
      & (points[:, 1] >= y0 - 0.02)
      & (points[:, 1] <= y1 + 0.02)
      & (distance <= 0.03)
    )
  x0, x1, y0, y1, z = bounds
  return (
    (points[:, 0] >= x0 - 0.02)
    & (points[:, 0] <= x1 + 0.02)
    & (points[:, 1] >= y0 - 0.02)
    & (points[:, 1] <= y1 + 0.02)
    & (np.abs(points[:, 2] - z) <= 0.03)
  )


def score_trial(rows, spec, geometry):
  if not rows or not rows[-1]["end"]:
    return {"status": "incomplete", "controls": len(rows)}
  if [row["tick"] for row in rows] != list(range(len(rows))):
    return {"status": "invalid", "reason": "noncontinuous control clock"}
  values = np.asarray(
    [[*row["velocity"], *row["angular"], row["tilt"], *row["command"]] for row in rows]
  )
  numeric = [
    value
    for row in rows
    for key in ("action", "received_action", "raw_action", "processed_targets", "root")
    for value in row[key]
  ]
  numeric.extend(
    value for row in rows for point in row["contact_points"] for value in point
  )
  witnessed = all(
    row["action"] == row["received_action"] == row["raw_action"]
    and np.allclose(
      np.asarray(row["command"]) * [2, 2, 0.25],
      row["history_command"],
      atol=1e-7,
      rtol=1e-6,
    )
    for row in rows
  )
  if not np.isfinite(values).all() or not np.isfinite(numeric).all() or not witnessed:
    return {
      "status": "invalid",
      "reason": "nonfinite or action/command identity mismatch",
    }
  if any(row["native_failure"] for row in rows):
    return {"status": "fail", "reason": "native_failure", "controls": len(rows)}
  if len(rows) < 50:
    return {"status": "fail", "reason": "less_than_one_second_evidence"}
  tail = rows[-50:]
  stable = all(
    np.linalg.norm(row["velocity"]) <= 0.10
    and np.linalg.norm(row["angular"]) <= 0.10
    and row["tilt"] <= 0.35
    and sum(row["contacts"]) >= 2
    for row in tail
  )
  if geometry is None:
    if len(rows) != spec["duration"]:
      return {"status": "fail", "reason": "early_end"}
    # 两段运动均按请求方向单独计量，不让一次反转掩盖失败方向。
    moving = True
    target = np.asarray(spec["target"])
    windows = (
      ((100, 550, 1), (750, 1200, -1))
      if spec["course"] == "general"
      else ((75, 200, 1), (325, 450, -1))
    )
    for first, last, sign in windows:
      segment = rows[first:last]
      if target.any():
        for axis in np.flatnonzero(target):
          actual = np.asarray(
            [r["angular"][2] if axis == 2 else r["velocity"][axis] for r in segment]
          )
          moving &= bool(
            actual.mean() * math.copysign(1, target[axis] * sign)
            >= abs(target[axis]) * 0.5
          )
      else:
        moving &= all(
          np.linalg.norm(row["velocity"]) <= 0.10
          and np.linalg.norm(row["angular"]) <= 0.10
          for row in segment
        )
    if spec["course"] == "general":
      roots = np.asarray([r["root"] for r in rows])
      max_offset = np.abs(roots[:, :2]).max(axis=1)
      left_platform = bool((max_offset > 1.0).any())
      within_patch = bool((max_offset <= 4.0).all())
      moving &= left_platform and within_patch
    return {
      "status": "pass" if stable and moving else "fail",
      "stable_stop": stable,
      "both_motion_directions": moving,
      "controls": len(rows),
    }
  length, width = geometry["length"], geometry["width"]
  core = geometry.get("core")
  entered = False
  floor_valid = True
  ordered = 0
  touched = np.zeros(4, dtype=bool)
  for row in rows:
    p = np.asarray(row["contact_points"])
    contact = np.asarray(row["contacts"], dtype=bool)
    root = row["root"]
    inside = 0.75 < root[0] < length - 0.75
    entered |= inside
    if abs(root[1]) > width - 0.05:
      floor_valid = False
    floor_valid &= not row["lower_floor"]
    if geometry["surfaces"]:
      support = np.zeros(4, dtype=bool)
      for surface in geometry["surfaces"]:
        support |= in_surface(p, surface)
      apron = ((p[:, 0] < 0.75) | (p[:, 0] > length - 0.75)) & (np.abs(p[:, 2]) <= 0.03)
      if core is not None:
        in_core = (p[:, 0] > core[0]) & (p[:, 0] < core[1])
        floor_valid &= not bool((contact & in_core & ~support).any())
      if inside:
        floor_valid &= not bool((contact & ~(support | apron)).any())
    if ordered < len(geometry["treads"]):
      touched |= contact & in_surface(p, geometry["treads"][ordered])
      if touched[:2].any() and touched[2:].any():
        ordered += 1
        touched[:] = False
  exit_stable = (
    stable
    and all(row["tilt"] < 0.20 for row in tail)
    and all(
      all(row["contacts"])
      and all(
        length - 0.75
        < (p[0] if spec["direction"] > 0 else length - p[0])
        < length + 0.65
        and abs(p[1]) < width - 0.02
        for p in row["contact_points"]
      )
      for row in tail
    )
  )
  treads_complete = ordered == len(geometry["treads"])
  passed = (
    entered
    and floor_valid
    and exit_stable
    and treads_complete
    and not rows[-1]["timeout"]
  )
  return {
    "status": "pass" if passed else "fail",
    "entered": entered,
    "valid_support": floor_valid,
    "exit_stable_one_second": exit_stable,
    "ordered_treads": ordered,
    "required_treads": len(geometry["treads"]),
    "controls": len(rows),
  }


def score(directory):
  directory = Path(directory)
  if (directory / "FAILED.json").exists() or not (directory / "RESULT.json").exists():
    raise ValueError("异常/未发布采集不能评分")
  manifest = json.loads((directory / "MANIFEST.json").read_text())
  if manifest["contract"] != CONTRACT or manifest["protocol"] != PROTOCOL:
    raise ValueError("Wolf评价合同/协议改变")
  result = json.loads((directory / "RESULT.json").read_text())
  if result["manifest_sha256"] != digest(manifest):
    raise ValueError("原始manifest摘要改变")
  scores = []
  for spec in manifest["trials"]:
    path = directory / spec["raw_file"]
    if (
      hashlib.sha256(path.read_bytes()).hexdigest()
      != result["raw_sha256"][spec["trial_id"]]
    ):
      raise ValueError("原始attempt摘要改变")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    scores.append(
      {
        "trial_id": spec["trial_id"],
        **score_trial(rows, spec, manifest["geometry"].get(spec["trial_id"])),
      }
    )
  return {
    "teacher_id": manifest["teacher_id"],
    "source_identity": manifest["source_identity"],
    "manifest_sha256": digest(manifest),
    "trials": scores,
    "complete": all(row["status"] != "incomplete" for row in scores),
    "qualified": False,
  }


def require_qualification(index, teacher_id, stage, max_level=None):
  if index is None:
    raise ValueError("阶段切换/正式学生训练需要独立原始评估索引")
  index = Path(index)
  directories = json.loads(index.read_text())["attempts"]
  required = requirements(stage)
  if not required:
    raise ValueError("无资格要求的阶段无需调用资格检查")
  selected_course = stage.split(":", 1)[-1]
  expected = {
    row["trial_id"]: row
    for course in required
    for seed in SEEDS
    for row in panel(course, seed, max_level if course == selected_course else None)
  }
  observed = {}
  identity = source_identity()
  for item in directories:
    path = index.parent / item
    report = score(path)
    manifest = json.loads((path / "MANIFEST.json").read_text())
    if report["teacher_id"] != teacher_id or report["source_identity"] != identity:
      raise ValueError("评估teacher或源码/资产/运行库不一致")
    for spec, outcome in zip(manifest["trials"], report["trials"], strict=True):
      key = spec["trial_id"]
      if key not in expected:
        continue
      raw_spec = {k: v for k, v in spec.items() if k != "raw_file"}
      if raw_spec != expected[key] or key in observed or outcome["status"] != "pass":
        raise ValueError("面板不符、重复或失败，不准晋级")
      # geometry由当前同源代码重新建立，拒绝手写简化接触面/少楼梯。
      if manifest["geometry"].get(key) != geometry_contract(spec):
        raise ValueError("原始地形证据合同不一致")
      observed[key] = outcome
  if observed.keys() != expected.keys():
    raise ValueError("三seed完整固定面板证据不足")
  return {
    "stage": stage,
    "max_level": max_level,
    "teacher_id": teacher_id,
    "trials": len(observed),
    "qualified": True,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("directory", type=Path)
  args = parser.parse_args()
  print(json.dumps(score(args.directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
