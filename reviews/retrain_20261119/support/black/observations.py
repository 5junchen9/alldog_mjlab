"""Append real terrain-ray evidence to the unchanged raw72 contract."""

from copy import deepcopy

import torch

from mjlab.managers import ObservationTermCfg
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg
from mjlab.tasks.velocity.config.black_rough.env_cfgs import OBSERVATION_TERMS

RAW72 = 72
RAYS = 187
WIDTH = 446
HEIGHT_CLIP_M = 5.0


def terrain_features_from_data(data, max_distance=5.0):
  """Signed hitZ-frameZ in meters, then mask; misses never imply groundZ=0."""
  distances = data.distances
  batch = distances.shape[0]
  if distances.shape != (batch, RAYS):
    raise ValueError("Expected one17x11 terrain frame, not foot-height rays")
  if data.hit_pos_w.shape != (batch, RAYS, 3) or data.normals_w.shape != (
    batch,
    RAYS,
    3,
  ):
    raise ValueError("Malformed terrain ray geometry")
  if data.frame_pos_w.shape != (batch, 1, 3) or data.frame_quat_w.shape != (
    batch,
    1,
    4,
  ):
    raise ValueError("Expected a single trunk sensor frame")
  if (
    not torch.isfinite(data.frame_pos_w).all()
    or not torch.isfinite(data.frame_quat_w).all()
    or (data.frame_quat_w.norm(dim=-1) < 1e-8).any()
    or (~torch.isfinite(distances)).all(dim=1).any()
  ):
    raise FloatingPointError(
      "Invalid or uninitialized terrain sensor; refuse policy input"
    )
  valid = (
    torch.isfinite(distances)
    & (distances >= 0)
    & (distances <= max_distance)
    & torch.isfinite(data.hit_pos_w).all(-1)
    & torch.isfinite(data.normals_w).all(-1)
    & (data.normals_w[..., 2] > 0)
  )
  relative = data.hit_pos_w[..., 2] - data.frame_pos_w[:, :, 2]
  heights = torch.where(valid, relative.clamp(-HEIGHT_CLIP_M, HEIGHT_CLIP_M), 0.0)
  return torch.cat((heights, valid.to(heights.dtype)), dim=-1)


def terrain_features(env):
  # ObservationManager calls terms once before it exists, solely to infer shape.
  # These masked-unknown extras are never used for a policy forward. Runtime
  # access below refuses uninitialized frame data; original72 is never filled.
  if not hasattr(env, "observation_manager"):
    return torch.zeros((env.num_envs, 2 * RAYS), device=env.device)
  sensor = env.scene["terrain_scan"]
  return terrain_features_from_data(sensor.data, sensor.cfg.max_distance)


def append_terrain(cfg, scanner_template=None):
  """Add exactly one374-term after the original72, without altering any base term."""
  for name in ("actor", "critic"):
    if tuple(cfg.observations[name].terms) != OBSERVATION_TERMS:
      raise ValueError("Original72 order/semantics must be established first")
  scanners = [s for s in cfg.scene.sensors if s.name == "terrain_scan"]
  if not scanners:
    if scanner_template is None:
      raise ValueError("Flat audit requires the real trunk terrain scanner")
    cfg.scene.sensors = (*cfg.scene.sensors, deepcopy(scanner_template))
    scanners = [cfg.scene.sensors[-1]]
  sensor = scanners[0]
  if (
    len(scanners) != 1
    or not isinstance(sensor, RayCastSensorCfg)
    or not isinstance(sensor.pattern, GridPatternCfg)
    or sensor.pattern.size != (1.6, 1.0)
    or sensor.pattern.resolution != 0.1
    or sensor.ray_alignment != "yaw"
    or sensor.include_geom_groups != (0,)
    or sensor.max_distance != 5.0
  ):
    raise ValueError("Real existing187-ray contract differs")
  for name in ("actor", "critic"):
    cfg.observations[name].terms["terrain_height_and_mask"] = ObservationTermCfg(
      func=terrain_features, scale=1.0
    )
  return cfg


def validate_raw446(raw):
  if raw.ndim != 2 or raw.shape[1] != WIDTH or not torch.isfinite(raw).all():
    raise FloatingPointError("Expected finite real raw72+height187+mask187")
  mask = raw[:, RAW72 + RAYS :]
  if not ((mask == 0) | (mask == 1)).all():
    raise ValueError("Terrain mask must be binary before normalization")
  if (raw[:, RAW72 : RAW72 + RAYS][mask == 0] != 0).any():
    raise ValueError("Missing heights require zero placeholder plus mask0")
