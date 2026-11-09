"""Training-only two-scale ray scans; these are NOT camera/depth-image inputs."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers import ManagerTermBase, ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

from .symmetry import left_right_mirror

TEACHER_LOCAL_GROUP = "terrain_teacher_local"
TEACHER_GLOBAL_GROUP = "terrain_teacher_global"
GLOBAL_SENSOR = "methods_global_terrain_scan"


def encode_teacher_scan(heights: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
  """Bound heights and retain an explicit validity mask for misses/nonfinites."""
  valid = (distances >= 0) & torch.isfinite(distances) & torch.isfinite(heights)
  bounded = torch.where(valid, heights, torch.zeros_like(heights)).clamp(-2, 2) / 2
  return torch.cat((bounded, valid.to(heights.dtype)), dim=-1)


class TerrainTeacherScan(ManagerTermBase):
  """Cache robot-relative ray positions; never copy world coordinates to actors."""

  def __init__(self, cfg, env) -> None:
    super().__init__(env)
    sensor = env.scene[cfg.params["sensor_name"]]
    offsets, _ = sensor.cfg.pattern.generate_rays(None, str(self.device))
    self.xy = (offsets[:, :2].T / 2.0).flatten()

  def __call__(self, env, sensor_name: str) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    encoded = encode_teacher_scan(
      mdp.height_scan(env, sensor_name=sensor_name), sensor.data.distances
    )
    return torch.cat((self.xy.expand(encoded.shape[0], -1), encoded), dim=-1)


def configure_terrain_teacher_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add privileged groups without modifying actor or critic observation layout."""
  if not any(sensor.name == "terrain_scan" for sensor in cfg.scene.sensors):
    raise ValueError("Terrain teacher requires the Wolf rough task's local scan")
  if any(sensor.name == GLOBAL_SENSOR for sensor in cfg.scene.sensors):
    raise ValueError("Terrain teacher observations already configured")
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    RayCastSensorCfg(
      name=GLOBAL_SENSOR,
      frame=ObjRef(type="body", name="base_link", entity="robot"),
      ray_alignment="yaw",
      pattern=GridPatternCfg(size=(3.2, 2.0), resolution=0.4),
      max_distance=5.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      debug_vis=False,
    ),
  )
  for group, sensor in (
    (TEACHER_LOCAL_GROUP, "terrain_scan"),
    (TEACHER_GLOBAL_GROUP, GLOBAL_SENSOR),
  ):
    cfg.observations[group] = ObservationGroupCfg(
      terms={
        "scan": ObservationTermCfg(
          func=TerrainTeacherScan, params={"sensor_name": sensor}
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )


def methods_left_right_mirror(env, obs, actions):
  """Mirror actor/critic groups only; teacher groups never enter actor losses.

  The training-only scans are not used by the actor at all. Preserve their shape
  to satisfy RSL-RL's augmented TensorDict contract. Auxiliary training reads the
  original rollout, never these augmented minibatches.
  """
  if obs is None:
    return left_right_mirror(env, obs, actions)
  base = obs.exclude(TEACHER_LOCAL_GROUP, TEACHER_GLOBAL_GROUP)
  mirrored, mirrored_actions = left_right_mirror(env, base, actions)
  assert mirrored is not None
  for key in (TEACHER_LOCAL_GROUP, TEACHER_GLOBAL_GROUP):
    if key in obs:
      mirrored[key] = torch.cat((obs[key], obs[key]), dim=0)
  return mirrored, mirrored_actions
