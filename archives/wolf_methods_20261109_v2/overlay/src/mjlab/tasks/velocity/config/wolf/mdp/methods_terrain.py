"""Wolf adaptations of support coverage and reduced-model reference shaping.

These are original project adapters, not copies of BeamDojo/PLANC code. Wolf is
wheel-legged: coverage is measured over a small wheel contact patch, not a sole.
The reference is constant-height LIPM using base COM as a centroidal proxy; it is
not PLANC's hybrid MLIP planner and provides no full-robot stability guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import mujoco
import numpy as np
import torch

from mjlab.managers import ManagerTermBase
from mjlab.sensor import GridPatternCfg
from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.utils.lab_api.math import quat_apply


@dataclass
class ElevatedGridPatternCfg(GridPatternCfg):
  elevation: float = 0.35

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 2] += self.elevation
    return offsets, directions


@dataclass(kw_only=True)
class WolfSupportTerrainCfg(SubTerrainCfg):
  """Matched soft/hard target geometry; only physical collision changes.

  Target surfaces remain in raycast group 0 in both stages. Soft stage adds a
  flat group-3 collision floor and disables target contacts. Hard stage enables
  target contacts over a lowered catch floor. Switch by rebuilding the scene,
  never by mutating collision semantics during a rollout.
  """

  soft: bool = True
  kind: str = "stones"
  gap_depth: float = 0.4

  def function(self, difficulty, spec, rng):
    if not 0 <= difficulty <= 1 or self.kind not in ("stones", "beams"):
      raise ValueError("Invalid support terrain difficulty/kind")
    body = spec.body("terrain")
    geometries = []
    length, width = self.size
    cx, cy = length / 2, width / 2

    def box(x, y, sx, sy, top, *, target=True):
      bottom = -self.gap_depth - 0.1
      depth = top - bottom
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x, y, (top + bottom) / 2),
        size=(sx / 2, sy / 2, depth / 2),
        group=0 if target else 3,
        contype=int(not target or not self.soft),
        conaffinity=int(not target or not self.soft),
      )
      geometries.append(TerrainGeometry(geom=geom))

    # A visible deep floor marks unsupported space for all target ray sensors.
    box(cx, cy, length, width, -self.gap_depth)
    if self.soft:
      box(cx, cy, length, width, 0.0, target=False)
    box(cx, cy, 2.0, width, 0.0)
    if self.kind == "beams":
      beam_width = 0.9 - 0.35 * difficulty
      box(cx, cy, length, beam_width, 0.0)
    else:
      spacing = 0.55 + 0.1 * difficulty
      stone_size = 0.52 - 0.16 * difficulty
      for x in np.arange(0.3, length, spacing):
        if abs(x - cx) < 0.9:
          continue
        for y in np.arange(0.3, width, spacing):
          top = float(rng.uniform(0.0, 0.06 * difficulty))
          box(x, y, stone_size, stone_size, top)
    return TerrainOutput(
      origin=np.array((cx, cy, 0.0)),
      geometries=geometries,
    )


def wheel_support_coverage(
  distances, hit_heights, wheel_bottoms, origin_z, *, soft, tolerance=0.05
):
  """Fraction of valid support samples [B,4,S] -> [B,4]."""
  if distances.ndim != 3 or distances.shape != hit_heights.shape:
    raise ValueError("Expected matching [env, wheel, sample] ray data")
  finite = torch.isfinite(hit_heights) & torch.isfinite(distances)
  valid = finite & (distances >= 0)
  if soft:
    # Soft-stage wheels touch the hidden plane, not ghost target elevations.
    valid &= hit_heights >= origin_z[:, None, None] - 0.10
  else:
    valid &= (hit_heights - wheel_bottoms[..., None]).abs() <= tolerance
  return valid.float().mean(-1)


class WheelPlacementReward(ManagerTermBase):
  """Continuous loaded-wheel penalty; sparse at support edges/gaps.

  Publishing the separately scaled contribution avoids recomputing rewards after
  reset and allows the second critic to see exactly the placement component.
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    self.weight = cfg.weight
    self.scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0

  def __call__(
    self,
    env,
    sensor_name,
    contact_sensor,
    asset_cfg,
    soft,
    wheel_radius=0.08,
    force_threshold=1.0,
  ):
    ray = env.scene[sensor_name].data
    wheels = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids]
    n = wheels.shape[1]
    coverage = wheel_support_coverage(
      ray.distances.reshape(env.num_envs, n, -1),
      ray.hit_pos_w[..., 2].reshape(env.num_envs, n, -1),
      wheels[..., 2] - wheel_radius,
      env.scene.terrain.env_origins[:, 2],
      soft=soft,
    )
    contacts = env.scene[contact_sensor].data
    loaded = contacts.force.norm(dim=-1) > force_threshold
    reward = -((1.0 - coverage) * loaded).sum(-1) / loaded.sum(-1).clamp_min(1)
    env.extras["wolf_sparse_reward"] = reward * self.weight * self.scale
    env.extras["log"]["Methods/wheel_support_coverage"] = (
      coverage * loaded
    ).sum() / loaded.sum().clamp_min(1)
    env.extras["log"]["Methods/loaded_wheel_fraction"] = loaded.float().mean()
    return reward


def project_to_support_hull(point, contacts, loaded):
  """Project 2-D points onto the convex hull of up to four loaded contacts."""
  if contacts.shape[1:] != (4, 2) or loaded.shape != contacts.shape[:2]:
    raise ValueError("Expected four planar contacts and a corresponding mask")
  edge_candidates = []
  edge_distances = []
  for a, b in combinations(range(4), 2):
    start = contacts[:, a]
    edge = contacts[:, b] - start
    t = ((point - start) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-9)
    candidate = start + t.clamp(0, 1)[:, None] * edge
    distance = (point - candidate).square().sum(-1)
    distance = torch.where(loaded[:, a] & loaded[:, b], distance, torch.inf)
    edge_candidates.append(candidate)
    edge_distances.append(distance)
  candidates = torch.stack(edge_candidates, dim=1)
  best = torch.stack(edge_distances, dim=1).argmin(-1)
  projection = candidates[torch.arange(point.shape[0], device=point.device), best]
  inside = torch.zeros(point.shape[0], dtype=torch.bool, device=point.device)
  for a, b, c in combinations(range(4), 3):
    v0 = contacts[:, b] - contacts[:, a]
    v1 = contacts[:, c] - contacts[:, a]
    v2 = point - contacts[:, a]
    determinant = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    denom = torch.where(determinant.abs() > 1e-9, determinant, 1.0)
    u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / denom
    v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / denom
    inside |= (
      loaded[:, a]
      & loaded[:, b]
      & loaded[:, c]
      & (determinant.abs() > 1e-9)
      & (u >= -1e-6)
      & (v >= -1e-6)
      & (u + v <= 1 + 1e-6)
    )
  return torch.where(inside[:, None], point, projection)


def lipm_reference_step(
  position,
  velocity,
  desired_velocity,
  contacts,
  loaded,
  height,
  dt,
  *,
  friction=0.3,
  gain=2.0,
):
  """One exact constant-height LIPM step with support/friction feasibility flag.

  Projection does not make an infeasible state feasible: outward accelerations
  above the conservative friction bound are flagged, never called certified.
  """
  if dt <= 0 or friction <= 0:
    raise ValueError("dt and friction must be positive")
  h = height.clamp(0.2, 0.8)
  acceleration = gain * (desired_velocity - velocity)
  acceleration *= (friction * 9.81 / acceleration.norm(dim=-1).clamp_min(1e-8)).clamp(
    max=1
  )[:, None]
  zmp = project_to_support_hull(
    position - acceleration * h[:, None] / 9.81, contacts, loaded
  )
  actual_acceleration = (position - zmp) * (9.81 / h[:, None])
  feasible = (
    (loaded.sum(-1) >= 3)
    & (height >= 0.2)
    & (height <= 0.8)
    & (actual_acceleration.norm(dim=-1) <= friction * 9.81 + 1e-5)
  )
  omega = (9.81 / h).sqrt()[:, None]
  ch, sh = torch.cosh(omega * dt), torch.sinh(omega * dt)
  next_position = zmp + (position - zmp) * ch + velocity * sh / omega
  next_velocity = (position - zmp) * omega * sh + velocity * ch
  return next_position, next_velocity, feasible, zmp


class LipmClfReward(ManagerTermBase):
  """Receding reduced-model reference and finite-difference CLF-like shaping.

  State is reset per environment. Only near-level >=3-contact support is used.
  The base COM is a proxy, friction is conservative, and angular momentum,
  actuator limits and stepping dynamics are not modeled. This is shaping, not
  a stability certificate, whole-body feasible trajectory, or safety filter.
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    self.reference_position = torch.zeros(self.num_envs, 2, device=self.device)
    self.reference_velocity = torch.zeros_like(self.reference_position)
    self.previous_v = torch.zeros(self.num_envs, device=self.device)
    self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.initialized[ids] = False
    self.previous_v[ids] = 0

  def __call__(
    self,
    env,
    contact_sensor,
    command_name,
    alpha=1.0,
    position_scale=0.3,
    velocity_scale=1.0,
    friction=0.3,
  ):
    data = env.scene["robot"].data
    contact = env.scene[contact_sensor].data
    loaded = (contact.found > 0) & (contact.force.norm(dim=-1) > 1.0)
    positions = contact.pos
    current = data.root_com_pos_w[:, :2]
    velocity = data.root_com_lin_vel_w[:, :2]
    ref = torch.where(self.initialized[:, None], self.reference_position, current)
    ref_vel = torch.where(self.initialized[:, None], self.reference_velocity, velocity)
    value = ((current - ref) / position_scale).square().sum(-1)
    value += ((velocity - ref_vel) / velocity_scale).square().sum(-1)
    residual = (value - self.previous_v) / env.step_dt + alpha * self.previous_v
    command = env.command_manager.get_command(command_name)
    cmd3 = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
    desired = quat_apply(data.root_link_quat_w, cmd3)[:, :2]
    support_z = (positions[..., 2] * loaded).sum(-1) / loaded.sum(-1).clamp_min(1)
    height = data.root_com_pos_w[:, 2] - support_z
    next_pos, next_vel, feasible, _ = lipm_reference_step(
      ref,
      ref_vel,
      desired,
      positions[..., :2],
      loaded,
      height,
      env.step_dt,
      friction=friction,
    )
    min_z = positions[..., 2].masked_fill(~loaded, torch.inf).amin(-1)
    max_z = positions[..., 2].masked_fill(~loaded, -torch.inf).amax(-1)
    feasible &= max_z - min_z <= 0.05
    # Large drift invalidates the local reduced-model approximation.
    feasible &= (current - ref).norm(dim=-1) <= 0.5
    active = self.initialized & feasible
    cost = torch.where(active, residual.clamp_min(0) + 0.1 * value, 0.0)
    self.reference_position.copy_(torch.where(feasible[:, None], next_pos, current))
    self.reference_velocity.copy_(torch.where(feasible[:, None], next_vel, velocity))
    self.previous_v.copy_(torch.where(feasible, value, 0.0))
    self.initialized.copy_(feasible)
    env.extras["log"]["Methods/lipm_reference_feasible"] = feasible.float().mean()
    env.extras["log"]["Methods/clf_violation"] = cost.mean()
    return cost


def support_base_height_error(env, contact_sensor, target_height=0.40):
  """Height above loaded support, avoiding low gap rays biasing the base down."""
  contact = env.scene[contact_sensor].data
  loaded = (contact.found > 0) & (contact.force.norm(dim=-1) > 1.0)
  support = (contact.pos[..., 2] * loaded).sum(-1) / loaded.sum(-1).clamp_min(1)
  origin = env.scene.terrain.env_origins[:, 2]
  support = torch.where(loaded.any(-1), support, origin)
  return (
    env.scene["robot"].data.root_link_pos_w[:, 2] - support - target_height
  ).square()


def below_support_surface(env, min_base_height=0.18):
  """Falling to the catch floor is failure, not an alternative route."""
  relative_height = (
    env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
  )
  return relative_height < min_base_height
