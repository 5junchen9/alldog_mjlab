"""Bounded low-stair route preparation; independent history and random stream."""

import copy
import math
import random
from typing import TypedDict

from support.black.grid_course import tuple_tree


class RoutePolicy(TypedDict):
  version: int
  rough_birth_fraction: float
  max_height_level: int
  path_degrees: list[int]
  spawn_x_jitter_m: float
  mid_y_jitter_m: float
  birth_yaw_jitter_degrees: float
  reference_half_length_m: float
  reference_half_width_m: float
  minimum_side_margin_m: float
  maximum_cross_track_m: float
  maximum_heading_error_degrees: float
  private_seed: int
  minimum_trials: int
  window: int
  success_fraction: float


POLICY: RoutePolicy = dict(
  version=1,
  rough_birth_fraction=0.10,
  max_height_level=1,
  path_degrees=[0, -1, 1],
  spawn_x_jitter_m=0.02,
  mid_y_jitter_m=0.02,
  birth_yaw_jitter_degrees=2.0,
  reference_half_length_m=0.50,
  reference_half_width_m=0.30,
  minimum_side_margin_m=0.02,
  maximum_cross_track_m=0.08,
  maximum_heading_error_degrees=10.0,
  private_seed=191126001,
  minimum_trials=32,
  window=64,
  success_fraction=0.90,
)


def cell(h, a, d):
  if (
    any(type(x) is not int for x in (h, a, d))
    or h not in (0, 1)
    or a not in (0, -1, 1)
    or d not in (-1, 1)
  ):
    raise ValueError("Unknown exact path-angle cell")
  return f"{h}/{a}/{d}"


def mastered(v):
  return len(v) >= 32 and sum(v) / len(v) >= 0.9


def geometry(length, width, angle, *, mid_y=0.0):
  if (
    not all(math.isfinite(x) for x in (length, width, angle, mid_y))
    or length <= 0
    or width <= 0
  ):
    raise ValueError("Finite positive route geometry required")
  alpha = math.radians(angle)
  if abs(alpha) >= math.pi / 2:
    raise ValueError("Route cannot run across step normals")
  heading = abs(alpha) + math.radians(POLICY["maximum_heading_error_degrees"])
  lateral = POLICY["reference_half_width_m"] * math.cos(heading) + POLICY[
    "reference_half_length_m"
  ] * math.sin(heading)
  center = (
    abs(mid_y)
    + (length / 2 + 0.70) * abs(math.tan(alpha))
    + POLICY["maximum_cross_track_m"]
  )
  margin = width - center - lateral
  # The entry feet remain before the first riser, whose distance is0.75m.
  initial = abs(alpha) + math.radians(POLICY["birth_yaw_jitter_degrees"])
  forward = POLICY["reference_half_length_m"] * math.cos(initial) + POLICY[
    "reference_half_width_m"
  ] * math.sin(initial)
  entry = 0.75 - POLICY["spawn_x_jitter_m"] - forward
  if margin < POLICY["minimum_side_margin_m"] or entry < 0.02:
    raise ValueError("No conservative corridor/entry footprint intersection")
  return dict(
    lateral_margin_m=margin,
    entry_clearance_m=entry,
    nominal_reference_only_not_all_action_envelope=True,
    full_root_interval=[-0.70, length + 0.70],
    path_degrees=angle,
  )


class SafeRoutes:
  def __init__(self):
    self.rng = random.Random(POLICY["private_seed"])
    self.cursor = 0
    self.history = {}
    self.routes = {}
    self.last_birth = {}
    self.records = 0
    self.last_allocation = {}
    self.measurements = {}

  def candidates(self, grid):
    rows = []
    for h in (0, 1):
      if not grid.both(h, 0):
        continue
      if h and (
        not grid.eligible(h, 0)
        or not all(
          mastered(self.history.get(cell(h - 1, a, d), []))
          for a in (0, -1, 1)
          for d in (-1, 1)
        )
      ):
        continue
      rows += [(h, a, d) for a in (0, -1, 1) for d in (-1, 1)]
    return rows

  def assign(self, ids, grid, lengths, widths, birth):
    legal = self.candidates(grid)
    chosen = []
    ordinary = []
    for i in ids:
      use = self.cursor % 10 == 9 and bool(legal)
      self.cursor += 1
      if not use:
        ordinary.append(i)
        continue
      # Cycle all available exact path/direction cells; private random jitter only.
      h, a, d = legal[(self.cursor // 10 - 1) % len(legal)]
      length = float(lengths[h])
      width = float(widths[h])
      dx = self.rng.uniform(-0.02, 0.02)
      mid = self.rng.uniform(-0.02, 0.02)
      jitter = self.rng.uniform(-2, 2)
      proof = geometry(length, width, a, mid_y=mid)
      x = (0 if d > 0 else length) + dx
      y = mid + math.tan(math.radians(a)) * (x - length / 2)
      yaw = (0 if d > 0 else math.pi) + math.radians(a + jitter)
      route = dict(
        kind="bounded_low_stair_path_v1",
        height=h,
        path_angle_degrees=a,
        direction=d,
        birth=birth,
        length_m=length,
        half_width_m=width,
        birth_x=x,
        birth_y=y,
        birth_yaw_radians=yaw,
        birth_yaw_jitter_degrees=jitter,
        mid_y=mid,
        nominal_grid_angle_degrees=0,
        first_contact=None,
        phase_seen=[False] * 3,
        phase_first=[None] * 3,
        geometry=proof,
      )
      self.routes[i] = route
      chosen.append((i, h, 0, d))
    for i in ordinary:
      self.routes.pop(i, None)
    old = grid.sample(len(ordinary))
    out = chosen + [(i, *row) for i, row in zip(ordinary, old, strict=True)]
    self.last_allocation = dict(
      total=len(ids), new_routes=len(chosen), legacy_routes=len(ordinary)
    )
    return out

  def finish(self, i, birth, strict_success):
    route = self.routes.get(i)
    if route is None:
      return False
    if route["birth"] != birth:
      raise ValueError("Episode route identity mismatch")
    previous = self.last_birth.get(i, -1)
    if birth < previous:
      raise ValueError("Out-of-order route episode")
    if previous == birth:
      return True
    key = cell(route["height"], route["path_angle_degrees"], route["direction"])
    values = self.history.setdefault(key, [])
    values.append(
      bool(strict_success and route["phase_seen"][0] and route["phase_seen"][2])
    )
    del values[:-64]
    measured = self.measurements.setdefault(key, [])
    measured.append(
      dict(
        route=copy.deepcopy(route),
        raw_strict_success=bool(strict_success),
        qualified_new_route_success=values[-1],
      )
    )
    del measured[:-64]
    self.last_birth[i] = birth
    self.records += 1
    return True

  def state_dict(self):
    return copy.deepcopy(
      dict(
        version=1,
        policy=POLICY,
        rng=self.rng.getstate(),
        cursor=self.cursor,
        history=self.history,
        measurements=self.measurements,
        routes={str(k): v for k, v in self.routes.items()},
        last_birth={str(k): v for k, v in self.last_birth.items()},
        records=self.records,
        last_allocation=self.last_allocation,
      )
    )

  @classmethod
  def from_state(cls, saved):
    if saved["version"] != 1 or saved["policy"] != POLICY:
      raise ValueError("Safe route schema/policy mismatch")
    obj = cls()
    for name in ("cursor", "records"):
      if type(saved[name]) is not int or saved[name] < 0:
        raise ValueError("Invalid route counter")
      setattr(obj, name, saved[name])
    for key, values in saved["history"].items():
      h, a, d = map(int, key.split("/"))
      assert cell(h, a, d) == key
      if (
        not isinstance(values, list)
        or len(values) > 64
        or any(type(x) is not bool for x in values)
      ):
        raise ValueError("Actual boolean route history required")
    obj.history = copy.deepcopy(saved["history"])
    obj.measurements = copy.deepcopy(saved["measurements"])
    if obj.measurements.keys() != obj.history.keys():
      raise ValueError("Missing measured angle evidence windows")
    for key, rows in obj.measurements.items():
      if any(
        type(r["raw_strict_success"]) is not bool
        or type(r["qualified_new_route_success"]) is not bool
        for r in rows
      ):
        raise ValueError("Measured success must be an actual boolean")
      if (
        len(rows) != len(obj.history[key])
        or [r["qualified_new_route_success"] for r in rows] != obj.history[key]
      ):
        raise ValueError("Measured angle evidence/window mismatch")
    obj.routes = {int(k): copy.deepcopy(v) for k, v in saved["routes"].items()}
    for i, r in obj.routes.items():
      if not 0 <= i < 4096:
        raise ValueError("Invalid route environment")
      cell(r["height"], r["path_angle_degrees"], r["direction"])
      if type(r["birth"]) is not int or r["birth"] < 0:
        raise ValueError("Invalid route birth")
      if len(r["phase_seen"]) != 3 or any(type(x) is not bool for x in r["phase_seen"]):
        raise ValueError("Invalid phase flags")
      geometry(
        r["length_m"], r["half_width_m"], r["path_angle_degrees"], mid_y=r["mid_y"]
      )
      endpoint = 0 if r["direction"] > 0 else r["length_m"]
      if abs(r["birth_x"] - endpoint) > 0.020001:
        raise ValueError("Unsafe saved entry x")
      expected_y = r["mid_y"] + math.tan(math.radians(r["path_angle_degrees"])) * (
        r["birth_x"] - r["length_m"] / 2
      )
      if abs(r["birth_y"] - expected_y) > 1e-7:
        raise ValueError("Saved route line mismatch")
      if any(
        (type(first) is not int or first < r["birth"]) if seen else first is not None
        for seen, first in zip(r["phase_seen"], r["phase_first"], strict=True)
      ):
        raise ValueError("Phase evidence lacks real control step")
      if any(r["phase_seen"]) != (r["first_contact"] is not None):
        raise ValueError("Missing actual first contact")
      if abs(r["mid_y"]) > 0.02 or abs(r["birth_yaw_jitter_degrees"]) > 2:
        raise ValueError("Unsafe saved jitter")
      if not all(
        math.isfinite(r[k])
        for k in ("birth_x", "birth_y", "birth_yaw_radians", "mid_y")
      ):
        raise ValueError("Nonfinite saved path")
    obj.last_birth = {int(k): v for k, v in saved["last_birth"].items()}
    if any(
      not 0 <= k < 4096 or type(v) is not int or v < 0
      for k, v in obj.last_birth.items()
    ):
      raise ValueError("Invalid route end token")
    obj.last_allocation = copy.deepcopy(saved["last_allocation"])
    obj.rng.setstate(tuple_tree(saved["rng"]))
    return obj
