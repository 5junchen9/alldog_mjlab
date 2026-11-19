"""Raw-hit446 extras and explicit gap semantics; no miss-to-ground substitution."""

from dataclasses import dataclass

import torch

from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.sensor.terrain_height_sensor import (
  TerrainHeightData,
  TerrainHeightSensor,
  TerrainHeightSensorCfg,
)
from support.black.observations import (
  terrain_features,
  terrain_features_from_data,
)

__all__ = [
  "terrain_features",
  "terrain_features_from_data",
  "raw_terrain_features",
  "GapAwareFootSensor",
  "GapAwareFootSensorCfg",
  "foot_clearance",
  "GAP_RAW72_CONTRACT",
  "GAP_SPECIALTY_SCHEMA",
  "classify_foot",
  "topology_support",
]


# Standard frozen RayCastSensor preserves -1 misses; there is no new raw bridge.
# Re-export the exact frozen function, including finite normals and normal_z>0.
raw_terrain_features = terrain_features_from_data
GAP_RAW72_CONTRACT = "raw72_gap_allmiss_maxrange_v1_UNADMITTED"
GAP_SPECIALTY_SCHEMA = "raw72gap_v1_height187_mask187_UNADMITTED"


def foot_clearance(raw, max_distance, num_frames=4):
  distances = raw.distances.reshape(len(raw.distances), num_frames, -1)
  hit_z = raw.hit_pos_w[..., 2].reshape_as(distances)
  normal_z = raw.normals_w[..., 2].reshape_as(distances)
  if not torch.isfinite(distances).all() or not torch.isfinite(raw.frame_pos_w).all():
    raise FloatingPointError("Invalid foot ray state")
  hit = (distances >= 0) & (distances <= max_distance)
  if (
    not torch.isfinite(raw.hit_pos_w.reshape(*distances.shape, 3)[hit]).all()
    or not torch.isfinite(raw.normals_w.reshape(*distances.shape, 3)[hit]).all()
  ):
    raise FloatingPointError("Invalid hit geometry in foot ray state")
  clearance = raw.frame_pos_w[:, :, 2, None] - hit_z
  clearance = torch.where(hit & (normal_z < 0), 0.0, clearance)
  # All-ray miss is explicitly saturated sensor range, not worldZ=0 ground.
  # This differs from the old no-hit fallback only; dynamic source admission
  # must verify the boundary. It must never be called strict raw72 replay.
  clearance = torch.where(hit, clearance, max_distance)
  return clearance.min(-1).values, hit.any(-1)


class GapAwareFootSensor(TerrainHeightSensor):
  def _compute_data(self):
    raw = RayCastSensor._compute_data(self)
    heights, self.support_ray_hit = foot_clearance(
      raw, self.cfg.max_distance, self.num_frames
    )
    return TerrainHeightData(**vars(raw), heights=heights)


@dataclass
class GapAwareFootSensorCfg(TerrainHeightSensorCfg):
  def build(self):
    return GapAwareFootSensor(self)


def classify_foot(
  *,
  sensor_finite,
  ray_hit,
  expected_support,
  hit_on_lower_floor,
  contact,
  fallen,
  outside,
):
  """Topology evidence is independent of raw-hit validity and contact."""
  if fallen:
    return "fall"
  if outside:
    return "out_of_bounds"
  if not sensor_finite:
    return "sensor_fault"
  if hit_on_lower_floor:
    return "lower_floor_failure" if contact else "lower_floor_seen"
  if not ray_hit:
    return "sensor_or_range_fault" if expected_support else "true_gap_no_hit"
  if expected_support:
    return "supported_contact" if contact else "support_surface_seen"
  return "non_support_hit"


def topology_support(feet, contacts, case, *, tolerance):
  """Use real horizontal box tops, never187-ray visibility as support.

  feet:(4,3) in original route-map coordinates; actual contact flags:(4,).
  Caller supplies an evidence-backed contact clearance tolerance at admission.
  """
  if feet.shape != (4, 3) or contacts.shape != (4,) or contacts.dtype != torch.bool:
    raise ValueError("Expected four real foot positions/contact flags")
  if (
    not torch.isfinite(feet).all()
    or not isinstance(tolerance, (int, float))
    or isinstance(tolerance, bool)
    or not 0 < tolerance < 1
  ):
    raise ValueError("Finite feet and explicit simulation support tolerance required")
  surfaces = case.get("support_surfaces")
  if not surfaces:
    raise RuntimeError("Gap/stake support topology not available")
  expected = torch.zeros(4, dtype=torch.bool, device=feet.device)
  supported = torch.zeros_like(expected)
  for xmin, xmax, ymin, ymax, top in surfaces:
    inside = (
      (feet[:, 0] >= xmin)
      & (feet[:, 0] <= xmax)
      & (feet[:, 1] >= ymin)
      & (feet[:, 1] <= ymax)
    )
    expected |= inside
    supported |= inside & ((feet[:, 2] - top).abs() <= tolerance) & contacts
  core = case["core"]
  inside_core = (feet[:, 0] > core[0]) & (feet[:, 0] < core[1])
  floor_contact = inside_core & contacts & ~supported
  return dict(
    expected_support=expected,
    supported_contact=supported,
    lower_floor_failure=floor_contact,
  )
