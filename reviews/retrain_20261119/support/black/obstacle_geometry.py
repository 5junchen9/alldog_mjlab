"""Exact historical collision subsets and parameterized layout, CPU-friendly.

Default dimensions come from TEST_MAP, never from the old0.2m wall fixture.
Rigid translation into a generated patch is explicit; mesh/contact/size stays.
"""

import copy
import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from support.black.terrain_cases import make_cases

SOURCE = Path(__file__).resolve().parent / "assets/obstacle_courses.xml"
GROUPS = {
  "stairs": (
    "stairs_04cm",
    "stairs_06cm",
    "stairs_08cm",
    "stairs_10cm",
    "stairs_12cm",
    "stairs_15cm",
    "stairs_20cm",
    "stairs_25cm",
    "stairs_30cm",
    "stairs_35cm",
    "stairs_40cm",
  ),
  "platform": tuple(
    f"platform_{n:02d}cm" for n in (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120)
  ),
  "high_wall": tuple(f"hurdle_{n}cm" for n in (10, 15, 20, 25, 30, 35, 40)),
  "slope": tuple(
    f"slope_{n:02d}deg" for n in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60)
  ),
  "broken_bridge": tuple(
    f"trench_{n}cm" for n in (10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180)
  ),
  "plum_stakes": ("plum", "plum_t2", "plum_t3", "plum_t4", "plum_t5"),
}


def selected_curriculum(task, rows=5):
  """Explicit five-row subset of the complete recorded family (no promotion)."""
  names = GROUPS[task]
  return tuple(names[round(i / (rows - 1) * (len(names) - 1))] for i in range(rows))


def source_sha():
  return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def patch_translation(case, size, world_position=(0, 0, 0)):
  """The same rigid translation for collision meshes, routes and box support."""
  return np.asarray(world_position) + [
    1.0 - (case["xmin"] - 0.75),
    size[1] / 2 - case["y"],
    0,
  ]


def patch_support_surfaces(case, size, world_position):
  dx, dy, dz = patch_translation(case, size, world_position)
  return [
    (a + dx, b + dx, c + dy, d + dy, z + dz)
    for a, b, c, d, z in case.get("support_surfaces", [])
  ]


def subset_xml(case_names: Sequence[str], *, include_floor=True):
  original = ET.parse(SOURCE).getroot()
  selected = []
  found = set()
  for geom in original.findall("./worldbody/geom"):
    name = geom.get("name", "")
    # Exact group boundary excludes plum_t2 from the base plum family.
    match = next(
      (
        case
        for case in sorted(case_names, key=lambda name: len(name), reverse=True)
        if name == case or name.startswith(case + "_")
      ),
      None,
    )
    if match == "plum" and name.startswith("plum_t"):
      match = None
    if ((include_floor and name == "floor") or match) and geom.get(
      "contype", "1"
    ) != "0":
      selected.append(geom)
      if match:
        found.add(match)
  if found != set(case_names):
    raise ValueError(
      "Missing historical collision cases:" + str(set(case_names) - found)
    )
  if not selected:
    raise ValueError("No historical collision geometry selected")
  target = ET.Element("mujoco", model="historical_specialty")
  ET.SubElement(target, "compiler", angle="radian")
  assets = ET.SubElement(target, "asset")
  meshes = {g.get("mesh") for g in selected if g.get("mesh")}
  materials = {g.get("material") for g in selected if g.get("material")}
  for asset in original.findall("./asset/mesh") + original.findall("./asset/material"):
    if asset.get("name") in meshes | materials:
      if asset.get("file"):
        raise ValueError("Unresolved external asset in selected collision subset")
      assets.append(copy.deepcopy(asset))
  textures = {m.get("texture") for m in assets if m.get("texture")}
  for texture in original.findall("./asset/texture"):
    if texture.get("name") in textures:
      if texture.get("file"):
        raise ValueError("Unresolved external texture")
      assets.append(copy.deepcopy(texture))
  body = ET.SubElement(target, "worldbody")
  for geom in selected:
    body.append(copy.deepcopy(geom))
  return ET.tostring(target, encoding="unicode")


def compile_cases(case_names):
  spec = mujoco.MjSpec.from_string(subset_xml(case_names))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data, make_cases(model)


def historical_case(name):
  model, data, cases = compile_cases([name])
  case = next(row for row in cases if row["name"] == name)
  return model, data, case


@dataclass(kw_only=True)
class SpecialtyTerrainCfg(SubTerrainCfg):
  task: str
  column: int = 0

  # Full recorded family is available; five generated rows are a preparation
  # selection, not a learned curriculum or a claim every height is reachable.
  def case_name(self, difficulty):
    choices = GROUPS[self.task]
    level = min(len(choices) - 1, int(float(difficulty) * len(choices)))
    return choices[level]

  def function(self, difficulty, spec, rng):
    del rng
    name = self.case_name(difficulty)
    _, _, case = historical_case(name)
    child = mujoco.MjSpec.from_string(subset_xml([name], include_floor=False))
    prefix = f"specialty_{len(spec.geoms)}_"
    # Rebase original route entry to a patch-local margin, preserving units.
    dx, dy, _ = patch_translation(case, self.size)
    if case["xmax"] + dx + 0.75 >= self.size[0]:
      raise ValueError("Historical route does not fit requested patch")
    # ContactSensor's secondary body is terrain. A worldbody frame would
    # render/raycast correctly while silently losing terrain contact matching.
    frame = spec.body("terrain").add_frame()
    spec.attach(child, prefix=prefix, frame=frame)
    geoms = [g for g in spec.geoms if g.name.startswith(prefix)]
    for geom in geoms:
      geom.pos = np.asarray(geom.pos) + [dx, dy, 0]
      geom.group = 0
    # Original TEST_MAP has a real floor at0 beneath gaps/stakes. A finite
    # patch floor preserves that height; it is classified as lower floor,
    # never as bridge/stake support.
    floor = spec.body("terrain").add_geom(
      name=prefix + "lower_floor",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=[self.size[0] / 2, self.size[1] / 2, -0.1],
      size=[self.size[0] / 2, self.size[1] / 2, 0.1],
      group=0,
    )
    geoms.append(floor)
    return TerrainOutput(
      origin=np.array([1.0, self.size[1] / 2, 0.0]),
      geometries=[TerrainGeometry(geom=g) for g in geoms],
    )


def route_from_case(case, *, reverse=False, entry_angle_degrees=0):
  direction = -1 if reverse else 1
  start = case["xmax"] + 0.75 if reverse else case["xmin"] - 0.75
  return dict(
    origin_xy=[start, case["y"]],
    heading_radians=np.pi if reverse else 0.0,
    entry_angle_degrees=float(entry_angle_degrees),
    direction=direction,
    gates=[0.75, case["xmax"] - case["xmin"] + 0.75],
    corridor_half_width=case["half_width"],
    source_case=case["name"],
    source_sha256=source_sha(),
    coordinate_transform="route projection only; original world geometry unchanged",
  )


def route_coordinates(route, xy):
  heading = route["heading_radians"]
  delta = np.asarray(xy) - route["origin_xy"]
  return np.array(
    [
      np.cos(heading) * delta[0] + np.sin(heading) * delta[1],
      -np.sin(heading) * delta[0] + np.cos(heading) * delta[1],
    ]
  )
