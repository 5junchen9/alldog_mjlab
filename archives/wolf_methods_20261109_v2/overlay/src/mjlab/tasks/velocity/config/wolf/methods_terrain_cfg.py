"""Opt-in support terrain and reduced-model teacher shaping configuration."""

from mjlab.asset_zoo.robots.wolf.wolf_constants import WOLF_WHEEL_BODY_NAMES
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.terrains import TerrainGeneratorCfg

from .env_cfg import WOLF_WHEEL_FORCE_CONTACT_SENSOR, WOLF_WHEEL_SLIP_CONTACT_SENSOR
from .mdp.methods_terrain import (
  ElevatedGridPatternCfg,
  LipmClfReward,
  WheelPlacementReward,
  WolfSupportTerrainCfg,
  below_support_surface,
  support_base_height_error,
)
from .mdp.rewards import base_height_l2_flat


def configure_terrain_teacher(cfg, *, soft_terrain, foot_placement=True, clf=True):
  """Configure a NEW rough/HIM cfg in place; existing task factories are untouched."""
  if cfg.scene.terrain.terrain_type != "generator":
    raise ValueError("Terrain teaching requires a rough Wolf configuration")
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(8.0, 4.0),
    num_rows=5,
    difficulty_range=(0.0, 1.0),
    border_width=1.0,
    sub_terrains={
      "stones": WolfSupportTerrainCfg(soft=soft_terrain, kind="stones"),
      "beams": WolfSupportTerrainCfg(soft=soft_terrain, kind="beams"),
    },
  )
  cfg.scene.terrain.max_init_terrain_level = 1
  # A soft-stage robot physically stands on z=0 despite target map gaps.
  if soft_terrain:
    cfg.rewards["base_height"].func = base_height_l2_flat
    cfg.rewards["base_height"].params = {"target_height": 0.40}
  else:
    cfg.rewards["base_height"].func = support_base_height_error
    cfg.rewards["base_height"].params = {
      "contact_sensor": WOLF_WHEEL_SLIP_CONTACT_SENSOR,
      "target_height": 0.40,
    }
    cfg.terminations["below_support_surface"] = TerminationTermCfg(
      func=below_support_surface,
    )
  if foot_placement:
    sensor_name = "wolf_wheel_support_scan"
    if any(s.name == sensor_name for s in cfg.scene.sensors):
      raise ValueError("Terrain teaching already configured")
    cfg.scene.sensors += (
      RayCastSensorCfg(
        name=sensor_name,
        frame=tuple(
          ObjRef(type="body", name=n, entity="robot") for n in WOLF_WHEEL_BODY_NAMES
        ),
        ray_alignment="world",
        pattern=ElevatedGridPatternCfg(size=(0.08, 0.04), resolution=0.02),
        max_distance=1.5,
        include_geom_groups=(0,),
        debug_vis=False,
      ),
    )
    cfg.rewards["wheel_support_placement"] = RewardTermCfg(
      func=WheelPlacementReward,
      weight=2.0,
      params={
        "sensor_name": sensor_name,
        "contact_sensor": WOLF_WHEEL_FORCE_CONTACT_SENSOR,
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=WOLF_WHEEL_BODY_NAMES, preserve_order=True
        ),
        "soft": soft_terrain,
      },
    )
  if clf:
    cfg.rewards["lipm_clf"] = RewardTermCfg(
      func=LipmClfReward,
      weight=-0.1,
      params={
        "contact_sensor": WOLF_WHEEL_SLIP_CONTACT_SENSOR,
        "command_name": "twist",
      },
    )
  cfg.viewer.reward_bar_max_terms = len(cfg.rewards)
  return cfg
