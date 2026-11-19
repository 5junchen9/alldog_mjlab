"""从原始测试地图碰撞几何提取独立项目边界。"""

import re

import mujoco
import numpy as np

GROUP_PATTERN = re.compile(
  r"^(platform_\d+cm|stairs_\d+cm|slope_\d+deg|trench_\d+cm|"
  r"plum(?:_t\d+)?|rough_t\d+|bridge_\d+cm|hurdle_\d+cm|cross_bar_\d+|"
  r"cross_slope_\d+deg|low_friction_\d+)(?:_|$)"
)


def outside_route_lane(case, position, feet):
  """窄桥只在桥体内限制机身；是否踩到底板由真实接触/箱顶检查判断。"""
  body_outside = abs(position[1] - case["y"]) > np.clip(
    case["half_width"] - 0.1, 0.25, 0.6
  )
  if case["category"] == "bridge":
    left, right, _ = case["core"]
    return left < position[0] < right and body_outside
  feet_outside = (np.abs(feet[:, 1] - case["y"]) > case["half_width"] - 0.05).any()
  return feet_outside or body_outside


def make_cases(model):
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  groups = {}
  for i in range(model.ngeom):
    name = model.geom(i).name.removeprefix("test/")
    match = GROUP_PATTERN.match(name)
    if match and model.geom_contype[i]:
      groups.setdefault(match[1], []).append(i)

  def bounds(ids):
    lows, highs = [], []
    for i in ids:
      rotation = data.geom_xmat[i].reshape(3, 3)
      if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[i]
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        vertices = (
          model.mesh_vert[start : start + count] @ rotation.T + data.geom_xpos[i]
        )
        lows.append(vertices.min(axis=0))
        highs.append(vertices.max(axis=0))
      else:
        center = data.geom_xpos[i] + rotation @ model.geom_aabb[i, :3]
        extent = np.abs(rotation) @ model.geom_aabb[i, 3:]
        lows.append(center - extent)
        highs.append(center + extent)
    return np.min(lows, axis=0), np.max(highs, axis=0)

  cases = []
  for name, ids in groups.items():
    low, high = bounds(ids)
    category = re.sub(r"_(?:\d+cm|\d+deg|\d+|t\d+)$", "", name)
    case = {
      "name": name,
      "category": category,
      "xmin": float(low[0]),
      "xmax": float(high[0]),
      "y": float((low[1] + high[1]) / 2),
      "half_width": float((high[1] - low[1]) / 2),
      "start_x": float(low[0] - 0.65),
      "timeout_s": min(150.0, float((high[0] - low[0] + 1.3) / 0.15 + 5)),
    }
    # 桥、沟槽、梅花桩不能把走到底部地板上算作通过。
    core_ids = []
    if category == "bridge":
      core_ids = [i for i in ids if model.geom(i).name.removeprefix("test/") == name]
    elif category == "trench":
      core_ids = [i for i in ids if model.geom(i).name.endswith(("_near", "_far"))]
    elif category == "plum":
      core_ids = [i for i in ids if re.search(r"_r[12]_\d+$", model.geom(i).name)]
    if core_ids:
      core_low, core_high = bounds(core_ids)
      case["core"] = [float(core_low[0]), float(core_high[0]), float(core_high[2])]
    if category in ("bridge", "trench", "plum"):
      # 真实可踩的水平箱顶，用于区分落在桥/桩上与掉到底部地板。
      surfaces, decks = [], []
      for geom_id in ids:
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
          continue
        lower, upper = bounds([geom_id])
        surfaces.append(
          [
            float(lower[0]),
            float(upper[0]),
            float(lower[1]),
            float(upper[1]),
            float(upper[2]),
          ]
        )
        name = model.geom(geom_id).name
        if name.endswith(("_entry", "_exit", "_near", "_far")):
          decks.append(
            [
              float((lower[0] + upper[0]) / 2),
              float((lower[1] + upper[1]) / 2),
              float(upper[2]),
            ]
          )
      case["support_surfaces"] = surfaces
      case["decks"] = sorted(decks)
    cases.append(case)
  return cases
