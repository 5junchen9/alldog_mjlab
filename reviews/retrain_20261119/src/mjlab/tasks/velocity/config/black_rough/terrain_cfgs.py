"""Five exact difficulty rows, native terrain geometry and safe spawn plateaus."""

from copy import deepcopy
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.heightfield_terrains import (
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  HfWaveTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxFlatTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

TERRAIN_NAMES = (
  "flat",
  "slope_up",
  "slope_down",
  "stairs_up",
  "stairs_down",
  "rough",
  "wave",
)
ROUGH_WEIGHTS = (0.15, 0.15, 0.15, 0.15, 0.20, 0.20)


def counts_for(total: int, weights: tuple[float, ...]) -> list[int]:
  raw = [total * x for x in weights]
  counts = [int(x) for x in raw]
  order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
  for i in order[: total - sum(counts)]:
    counts[i] += 1
  return counts


def domain_counts(total: int) -> list[int]:
  flat = round(total * 0.2)
  return [flat, *counts_for(total - flat, ROUGH_WEIGHTS)]


@dataclass(kw_only=True)
class DedicatedFlatCfg(SubTerrainCfg):
  """One 96m flat floor outside the rough grid; no neighboring rough leakage.

  All flat environments use row zero. Extra flat row origins are retained only
  for the common 5x7 map schema. The floor is created once, never overlapping
  several coplanar boxes, and is far from the rough hfields.
  """

  def function(self, difficulty, spec, rng):
    origin = np.array([self.size[0] / 2, -100.0, 0.0])
    existing = any(g.name == "black_v2_dedicated_flat" for g in spec.geoms)
    geometry = []
    if not existing:
      geom = spec.body("terrain").add_geom(
        name="black_v2_dedicated_flat",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[origin[0], origin[1], -0.1],
        size=[48.0, 48.0, 0.1],
      )
      geometry = [TerrainGeometry(geom=geom)]
    return TerrainOutput(origin=origin, geometries=geometry)


@dataclass(kw_only=True)
class SpawnPlateauCfg(SubTerrainCfg):
  """Native hfield with a central 2m plateau and 0.25m transition.

  The plateau is a geometry edit, not a reward or observation substitution.
  Native stairs/slopes already supply their own central platform.
  """

  inner: SubTerrainCfg = field(default_factory=BoxFlatTerrainCfg)
  rough_half_range_m: float | None = None

  def function(self, difficulty, spec, rng):
    if difficulty == 0:
      return BoxFlatTerrainCfg(size=self.size).function(0, spec, rng)
    inner = deepcopy(self.inner)
    inner.size = self.size
    output = inner.function(difficulty, spec, rng)
    for geometry in output.geometries:
      hfield = geometry.hfield
      if hfield is None:
        continue
      data = np.asarray(hfield.userdata).reshape(hfield.nrow, hfield.ncol)
      if self.rough_half_range_m is not None:
        # Native spline/rounding may overshoot noise_range. Bound the final
        # physical grid, with nominal zero at its pre-clamp midrange and a
        # zero-height central platform. Do not change the shared native class.
        half_range = self.rough_half_range_m * difficulty
        physical = data * hfield.size[2]
        physical -= 0.5 * (physical.min() + physical.max())
        physical = np.clip(physical, -half_range, half_range)
        size = list(hfield.size)
        size[2] = 2 * half_range
        hfield.size[:] = size
        assert geometry.geom is not None
        position = list(geometry.geom.pos)
        position[2] -= half_range
        geometry.geom.pos[:] = position
        data = (physical + half_range) / (2 * half_range)
      x = np.linspace(-self.size[0] / 2, self.size[0] / 2, hfield.nrow)
      y = np.linspace(-self.size[1] / 2, self.size[1] / 2, hfield.ncol)
      radius = np.maximum(np.abs(x[:, None]), np.abs(y[None, :]))
      plateau = (
        0.5
        if self.rough_half_range_m is not None
        else float(np.max(data[radius <= 1.0]))
      )
      blend = np.clip((radius - 1.0) / 0.25, 0, 1)
      data = plateau * (1 - blend) + data * blend
      hfield.userdata = data.astype(np.float32).flatten().tolist()
      assert geometry.geom is not None
      output.origin[2] = geometry.geom.pos[2] + plateau * hfield.size[2]
    return output


def terrain_config(seed: int = 112001) -> TerrainGeneratorCfg:
  terrains = {
    "flat": DedicatedFlatCfg(proportion=0.2),
    "slope_down": HfPyramidSlopedTerrainCfg(slope_range=(0.0, 0.2), platform_width=2.0),
    "slope_up": HfPyramidSlopedTerrainCfg(
      slope_range=(0.0, 0.2), platform_width=2.0, inverted=True
    ),
    "stairs_down": BoxPyramidStairsTerrainCfg(
      step_height_range=(0.0, 0.08),
      step_width=0.4,
      platform_width=2.0,
      border_width=0.25,
    ),
    "stairs_up": BoxInvertedPyramidStairsTerrainCfg(
      step_height_range=(0.0, 0.08),
      step_width=0.4,
      platform_width=2.0,
      border_width=0.25,
    ),
    "rough": SpawnPlateauCfg(
      rough_half_range_m=0.02,
      inner=HfRandomUniformTerrainCfg(
        noise_range=(-0.02, 0.02),
        noise_step=0.005,
        vertical_scale=0.005,
        horizontal_scale=0.1,
        downsampled_scale=0.2,
        scale_with_difficulty=True,
        border_width=0.25,
      ),
    ),
    "wave": SpawnPlateauCfg(
      inner=HfWaveTerrainCfg(
        amplitude_range=(0.0, 0.04), num_waves=2, border_width=0.25
      )
    ),
  }
  terrains = {name: terrains[name] for name in TERRAIN_NAMES}
  for name, weight in zip(TERRAIN_NAMES[1:], ROUGH_WEIGHTS, strict=True):
    terrains[name].proportion = 0.8 * weight
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=True,
    size=(8.0, 8.0),
    num_rows=5,
    num_cols=7,
    border_width=4.0,
    sub_terrains=terrains,
    color_scheme="none",
  )
