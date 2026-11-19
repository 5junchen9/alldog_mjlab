"""v2固定多seed资格面板及绝对指标；分项评分和总资格分开。"""

import hashlib
import json
import math
import struct
from typing import Any

from curriculum import EVALUATION_SEEDS, SETTLE_STEPS

SCHEMA = "shared_actor_evaluation_v2"
PROTOCOL_ID = "shared_actor_absolute_qualification_v2"
QUALIFICATION_ID = "all_cells_all_seeds_v2"
FLAT8 = {
  "forward": [0.3, 0.0, 0.0],
  "lateral_left": [0.0, 0.3, 0.0],
  "lateral_right": [0.0, -0.3, 0.0],
  "reverse": [-0.3, 0.0, 0.0],
  "reversal": [0.3, 0.0, 0.0],
  "stop": [0.3, 0.0, 0.0],
  "yaw_left": [0.0, 0.0, 0.5],
  "yaw_right": [0.0, 0.0, -0.5],
}
HEIGHTS_CM = (4, 6, 8, 10, 12, 15, 20, 25, 30)
PROTOCOL: dict[str, Any] = {
  "control_hz": 50,
  "batch_size": 32,
  "motion_fraction": 0.5,
  "park_mean_planar_max": 0.10,
  "park_mean_yaw_max": 0.15,
  "axis_settle_planar_max": 0.08,
  "axis_settle_yaw_max": 0.12,
  "axis_settle_tilt_max": 0.35,
  "axis_settle_contacts_min": 2,
  "axis_settle_steps": SETTLE_STEPS,
  "axis_response_steps": 10,
  "stair_ordered_treads": 12,
  "stair_strict_settle_steps": SETTLE_STEPS,
  "stair_linear_norm_max": 0.08,
  "stair_angular_norm_max": 0.12,
  "stair_tilt_max": 0.35,
  "qualification": QUALIFICATION_ID,
  "evaluation_seeds": EVALUATION_SEEDS,
  "required_trial_success_fraction": 1.0,
  "scope": "per_trial_score_then_recompute_complete_fixed_panel_for_admission",
}


def digest(value):
  encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(encoded).hexdigest()


def command(spec, tick):
  """动作k之前发布的命令；float32表示与原生command缓冲一致。"""
  target = list(spec["target"])
  if spec["program"] == "flat8":
    if tick >= 150:
      if spec["case"] == "reversal":
        target = [-0.3, 0.0, 0.0]
      elif spec["case"] == "stop":
        target = [0.0, 0.0, 0.0]
  elif spec["program"] == "axis":
    if tick < 100 or tick >= 600:
      target = [0.0, 0.0, 0.0]
    elif tick >= 350:
      target = [-value for value in target]
  else:
    raise ValueError("楼梯命令由原生路线控制，不使用平地命令表")
  return [struct.unpack("f", struct.pack("f", value))[0] for value in target]


def roster(suite, seed, heights_cm=HEIGHTS_CM, angles_deg=(-15, 0, 15)):
  """一次seed面板；批次、lane和实际环境seed在执行前固定。"""
  rows = []
  if suite == "flat8":
    for repeat in range(4):
      for case, target in FLAT8.items():
        rows.append(
          dict(
            kind="flat",
            program="flat8",
            case=case,
            target=target,
            orientation=0.0,
            horizon=400,
            repeat=repeat,
          )
        )
  elif suite == "axes":
    for axis in range(3):
      for magnitude in (0.10, 0.25):
        for sign in (-1, 1):
          for orientation in (0.0, math.pi):
            target = [0.0, 0.0, 0.0]
            target[axis] = sign * magnitude
            rows.append(
              dict(
                kind="flat",
                program="axis",
                target=target,
                orientation=orientation,
                horizon=900,
              )
            )
  elif suite == "stairs":
    if (
      not heights_cm
      or len(set(heights_cm)) != len(heights_cm)
      or any(value not in HEIGHTS_CM for value in heights_cm)
    ):
      raise ValueError("楼梯高度需从当前4–30cm路线中不重复选择")
    if (
      not angles_deg
      or len(set(angles_deg)) != len(angles_deg)
      or any(value not in (-15, 0, 15) for value in angles_deg)
    ):
      raise ValueError("入场角需从-15、0、15度中不重复选择")
    for height in heights_cm:
      for direction in (-1, 1):
        for angle in angles_deg:
          rows.append(
            dict(
              kind="stairs",
              program="stairs",
              target=None,
              orientation=0.0,
              horizon=6000,
              height_cm=height,
              direction=direction,
              angle_deg=angle,
            )
          )
  else:
    raise ValueError("unknown evaluation suite")
  for index, row in enumerate(rows):
    row.update(
      trial_id=f"trial_{index:04d}",
      batch_id=index // 32,
      lane=index % 32,
      seed=seed + index // 32,
    )
  return rows


def padded_batch(rows):
  """原生命令实现需要32lane；补位有独立标识，不进入试验分母。"""
  if not 1 <= len(rows) <= 32:
    raise ValueError("每批需要1–32条登记试验")
  result: list[dict] = [dict(row, padding=False) for row in rows]
  for lane in range(len(rows), 32):
    result.append(dict(rows[0], trial_id=f"padding_{lane}", lane=lane, padding=True))
  return result
