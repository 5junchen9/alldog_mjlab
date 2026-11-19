"""Six isolated actual framework configurations; no environment/GPU launch."""

import copy
import math
from dataclasses import dataclass, fields

import torch

import support.black.passage_native as passage_native
from mjlab.envs import mdp as common
from mjlab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.tasks.velocity.config.black_rough import mdp as original_mdp
from mjlab.tasks.velocity.config.black_rough.terrain_cfgs import DedicatedFlatCfg
from mjlab.terrains import TerrainGeneratorCfg
from support.black.completion_value import CompletionValueLedger, verify_return_bound
from support.black.config import comprehensive_configs, configure_ppo_schedule
from support.black.exit_stability import ExitStabilityLedger
from support.black.flat_coverage import FlatCoverage, initialize_context, reset_context
from support.black.grid_course import POLICY as GRID_POLICY
from support.black.grid_course import GridCurriculum
from support.black.obstacle_geometry import (
  GROUPS,
  SpecialtyTerrainCfg,
  historical_case,
)
from support.black.progress_credit import ProgressCredit, ProgressDenseCost
from support.black.route_runtime import RouteRuntime
from support.black.safe_route import SafeRoutes
from support.black.sensors import (
  terrain_features,
)
from support.black.specialty_composition import (
  CompositionCommand,
  CompositionCommandCfg,
)
from support.black.stair_metrics import EpisodeMetrics
from support.black.stair_progress import StairCatalog, StairLedger
from support.black.typed_runtime import require_terrain
from support.black.vector_ledger import VectorLedger

ENTRY_STAGES = (
  (0,),
  (0, -15, 15),
  (0, -15, 15, -30, 30),
  (0, -15, 15, -30, 30, -45, 45),
)


def terminal_evidence(
  *,
  foot_progress,
  foot_lateral,
  exit_progress,
  exit_apron_end,
  corridor_half_width,
  feet_z,
  contacts,
  linear_velocity,
  angular_velocity,
  projected_gravity,
  parameters,
):
  """Four feet on the exit apron plus explicit simulation stability bounds.

  No hardware thresholds/defaults are invented here. Bounds need simulation
  admission evidence; tests supply named synthetic fixture values only.
  """
  if (
    any(
      value.shape != (4,) for value in (foot_progress, foot_lateral, feet_z, contacts)
    )
    or contacts.dtype != torch.bool
  ):
    raise ValueError("Terminal support requires four actual feet")
  tensors = (
    foot_progress,
    foot_lateral,
    feet_z,
    linear_velocity,
    angular_velocity,
    projected_gravity,
  )
  if not all(torch.isfinite(value).all() for value in tensors):
    return False
  return bool(
    (foot_progress > exit_progress).all()
    and (foot_progress <= exit_apron_end).all()
    and (foot_lateral.abs() <= corridor_half_width).all()
    and contacts.all()
    and (feet_z.abs() <= parameters["finish_support_z_tolerance"]).all()
    and linear_velocity.norm() <= parameters["finish_max_linear_speed"]
    and angular_velocity.norm() <= parameters["finish_max_angular_speed"]
    and torch.acos((-projected_gravity[2]).clamp(-1, 1))
    <= parameters["finish_max_tilt_radians"]
  )


class SpecialtyCommand(CompositionCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    initialize_context(self)
    self.route_direction = torch.ones(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self.entry_angle = torch.zeros(self.num_envs, device=self.device)
    self.route_sample_step = torch.full(
      (self.num_envs,), -1, device=self.device, dtype=torch.long
    )

  def reset(self, env_ids):
    reset_context(self, env_ids)
    return super().reset(env_ids)

  def sample_route_entries(self, env_ids):
    ids = env_ids[~self.flat_mask[env_ids]]
    if getattr(self.cfg, "task", None) == "stairs":
      st = specialty_state(self._env)
      if st.grid is not None:
        assign_grid_routes(st, ids)
        return
    self.route_direction[ids] = (
      torch.randint(2, (len(ids),), device=self.device) * 2 - 1
    )
    state = specialty_state(self._env)
    angles = torch.tensor(ENTRY_STAGES[state.angle_stage], device=self.device)
    self.entry_angle[ids] = (
      angles[torch.randint(len(angles), (len(ids),), device=self.device)]
      * math.pi
      / 180
    )
    self.route_sample_step[ids] = self._env.common_step_counter

  def _resample_command(self, env_ids):
    before = self.vel_command_b[env_ids].clone()
    super()._resample_command(env_ids)
    if hasattr(self._env, "specialty446_state"):
      coverage = specialty_state(self._env).flat_coverage
      if coverage is not None:
        coverage.resample(self, env_ids, before, self._env.common_step_counter)
    ids = env_ids[~self.flat_mask[env_ids]]
    # Episode route survives speed resampling. No yaw-only stalls on an obstacle.
    self.target_command[ids] = 0
    self.target_command[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(
      0.15, 0.35
    )

  def _update_command(self, env_ids=None):
    super()._update_command(env_ids)
    if not hasattr(self, "route_direction"):
      return
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    ids = ids[~self.flat_mask[ids]]
    robot = self._env.scene["robot"]
    target = torch.where(self.route_direction[ids] > 0, 0.0, math.pi)
    if hasattr(self._env, "specialty446_state"):
      routes = specialty_state(self._env).routes
      if routes is not None:
        target = target + routes.angle[ids]
    error = torch.atan2(
      torch.sin(target - robot.data.heading_w[ids]),
      torch.cos(target - robot.data.heading_w[ids]),
    )
    speed = self.target_command[ids, 0]
    if hasattr(self._env, "specialty446_state"):
      st = specialty_state(self._env)
      length = st.lengths[require_terrain(self._env).terrain_levels[ids]]
      origin = self._env.scene.env_origins[ids]
      delta = robot.data.root_link_pos_w[ids, 0] - origin[:, 0]
      progress = torch.where(self.route_direction[ids] > 0, delta, length - delta)
      # Bring all feet past the exit before requesting a stable halt.
      speed = torch.where(progress >= length - 0.20, 0.0, speed)
    self.vel_command_b[ids, 0] = speed * error.cos()
    self.vel_command_b[ids, 1] = speed * error.sin()
    self.vel_command_b[ids, 2] = error.clamp(-0.35, 0.35)


@dataclass(kw_only=True)
class SpecialtyCommandCfg(CompositionCommandCfg):
  task: str
  course_parameters: dict | None = None
  passage_course_policy: dict | None = None

  def build(self, env):
    return SpecialtyCommand(self, env)


class SpecialtyState:
  def __init__(self, env):
    self.env = env
    self.command = env.command_manager.get_term("twist")
    self.parameters = copy.deepcopy(self.command.cfg.course_parameters)
    if not self.parameters:
      raise ValueError("Named simulation parameters required")
    self.cases = [historical_case(name)[2] for name in GROUPS[self.command.cfg.task]]
    self.lengths = torch.tensor(
      [c["xmax"] - c["xmin"] + 1.5 for c in self.cases], device=env.device
    )
    self.widths = torch.tensor([c["half_width"] for c in self.cases], device=env.device)
    self.failed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.finished = torch.zeros_like(self.failed)
    self.ledger = VectorLedger(env.num_envs, env.device, self.parameters)
    self.stair_catalog = (
      StairCatalog(env.sim.mj_model, env.device)
      if self.command.cfg.task == "stairs"
      else None
    )
    self.sparse_catalog = None
    if self.command.cfg.task in ("broken_bridge", "plum_stakes"):
      from support.black.sparse_support import SparseSupportCatalog

      self.sparse_catalog = SparseSupportCatalog(env.sim.mj_model, env.device)
    self.stair_ledger = StairLedger(env.num_envs, env.device)
    self.exit_stability = ExitStabilityLedger(
      env.num_envs, env.device, settle_steps=self.parameters["settle_steps"]
    )
    self.completion_value = CompletionValueLedger(env.num_envs, env.device)
    self.stair_counts = torch.tensor(
      (
        [self.stair_catalog.counts[name] for name in GROUPS[self.command.cfg.task]]
        if self.stair_catalog is not None
        else [0] * len(self.cases)
      ),
      device=env.device,
    )
    self.episode_metrics = EpisodeMetrics(env)
    self.last_step = -1
    self.height_level = 0
    self.angle_stage = 0
    self.history = {}
    self.grid = GridCurriculum() if self.command.cfg.task == "stairs" else None
    if self.grid is not None:
      if any(
        self.parameters[name] != GRID_POLICY[name]
        for name in ("minimum_trials", "window", "success_fraction")
      ):
        raise ValueError("Grid thresholds must equal the unchanged course parameters")
      self.history = self.grid.history
    self.routes = RouteRuntime(self) if self.grid is not None else None
    self.flat_coverage = FlatCoverage() if self.grid is not None else None
    self.progress_credit = ProgressCredit(
      env.num_envs, env.device, num_levels=len(self.cases)
    )
    self.promotions = []
    self.feet, _ = env.scene["robot"].find_sites(
      ("FL", "FR", "RR", "RL"), preserve_order=True
    )
    params = env.cfg.observations["actor"].terms["foot_contact"].params
    self.contact_name = params["sensor_name"]
    self.passage_course = None
    if self.command.cfg.passage_course_policy is not None:
      passage_native.init(self)

  def update(self):
    env = self.env
    if self.last_step == env.common_step_counter:
      return
    self.last_step = env.common_step_counter
    base = original_mdp.state(env)
    base.enforce_patch_bounds = False
    base.update()
    robot = env.scene["robot"]
    origin = env.scene.env_origins
    levels = env.scene.terrain.terrain_levels
    length = self.lengths[levels]
    width = self.widths[levels]
    direction = self.command.route_direction
    delta = robot.data.root_link_pos_w[:, 0] - origin[:, 0]
    progress = torch.where(direction > 0, delta, length - delta)
    positions = robot.data.site_pos_w[:, self.feet]
    footx = positions[:, :, 0] - origin[:, None, 0]
    foot_progress = torch.where(
      (direction > 0)[:, None], footx, length[:, None] - footx
    )
    footy = positions[:, :, 1] - origin[:, None, 1]
    # Retain the actual task/reward physical snapshot. Interval pushes happen
    # after reward and before reset; terminal logs must not read their new state.
    self.control_snapshot = {
      "root_local": (robot.data.root_link_pos_w - origin).clone(),
      "terrain_origin": origin.clone(),
      "root_velocity_world": robot.data.root_link_lin_vel_w.clone(),
      "actual_command": self.command.command.clone(),
      "foot_progress": foot_progress.clone(),
      "foot_local_z": (positions[:, :, 2] - origin[:, None, 2]).clone(),
      "progress": progress[:, None].clone(),
    }
    contact = env.scene[self.contact_name].data.found.reshape(env.num_envs, 4) > 0
    scan = env.scene["foot_height_scan"].data
    valid = torch.isfinite(scan.distances).all(-1)
    if valid.ndim > 1:
      valid = valid.flatten(1).all(-1)
    valid &= (scan.distances.reshape(env.num_envs, 4, -1) >= 0).any(-1).all(-1)
    sparse = None
    if self.sparse_catalog is not None:
      sparse = self.sparse_catalog.read(env.sim.data, env.num_envs)
      # 空隙没有射线命中是有效观测；只检查真实命中位置和法向。
      hit = scan.distances >= 0
      valid = torch.isfinite(scan.distances).flatten(1).all(-1)
      finite_hit = torch.isfinite(scan.hit_pos_w).all(-1) & torch.isfinite(
        scan.normals_w
      ).all(-1)
      valid &= (~hit | finite_hit).flatten(1).all(-1)
      contact = sparse["supported"]
    lateral = (robot.data.root_link_pos_w[:, 1] - origin[:, 1]).abs()
    bounds = (lateral <= width) & (progress >= -0.70) & (progress <= length + 0.70)
    rough = ~self.command.flat_mask
    base_failed = base.failed
    if sparse is not None:
      # 桥桩核心区的支撑由真实接触定义；高度射线不替代碰撞证据。
      reasons = base.reason.clone()
      reasons[:, -1] = False
      base_failed = torch.where(self.command.flat_mask, base.failed, reasons.any(-1))
      base_failed |= ~torch.isfinite(robot.data.root_link_pos_w).all(-1)
    self.failed.copy_(base_failed | (rough & (~valid | ~bounds)))
    if sparse is not None:
      self.failed |= rough & (sparse["lower_floor"].any(-1) | sparse["invalid"])

    if self.routes is not None:
      self.failed |= self.routes.violation(
        robot.data.root_link_pos_w,
        positions,
        length,
        width,
        direction,
        robot.data.heading_w,
      )
      self.routes.observe()
    self.event_snapshot = {
      "schema": "black-specialty-events-v1",
      "step": env.common_step_counter,
      "base_failed": base.failed.detach().clone(),
      "scan_invalid": (rough & ~valid).detach().clone(),
      "bounds_invalid": (rough & ~bounds).detach().clone(),
      "route_bits": (
        None if self.routes is None else self.routes.guard_bits.detach().clone()
      ),
      "combined_failed": self.failed.detach().clone(),
    }
    p = self.parameters
    settled = (foot_progress >= (length - 0.75)[:, None]).all(-1)
    settled &= (foot_progress <= (length + 0.70)[:, None]).all(-1)
    settled &= (footy.abs() <= width[:, None]).all(-1)
    settled &= (
      (positions[:, :, 2] - origin[:, None, 2]).abs() <= p["finish_support_z_tolerance"]
    ).all(-1)
    safe_full_exit = settled.clone()
    sensor = env.scene[self.contact_name].data
    contact_valid = torch.isfinite(sensor.found.reshape(env.num_envs, 4)).all(-1)
    contact_valid &= (sensor.found.reshape(env.num_envs, 4) >= 0).all(-1)
    if sensor.force is None:
      contact_valid &= False
    else:
      contact_valid &= torch.isfinite(sensor.force).flatten(1).all(-1)
    safe_full_exit &= contact_valid
    settled &= contact.all(-1)
    settled &= (
      robot.data.root_link_lin_vel_w.norm(dim=-1) <= p["finish_max_linear_speed"]
    )
    settled &= (
      robot.data.root_link_ang_vel_w.norm(dim=-1) <= p["finish_max_angular_speed"]
    )
    settled &= (
      torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
      <= p["finish_max_tilt_radians"]
    )
    gates = torch.stack((torch.full_like(length, 0.75), length - 0.75), -1)
    self.ledger.prepare(
      env.common_step_counter,
      env.step_dt,
      progress,
      gates,
      rough,
      contact.sum(-1) >= 2,
      settled,
      self.failed,
      valid,
      exit_safe=safe_full_exit,
    )
    self.failed |= rough & self.ledger.invalid
    self.finished.copy_(
      rough
      & self.ledger.closed
      & (self.ledger.settled_steps >= p["settle_steps"])
      & ~self.failed
    )
    self.exit_stability.prepare(
      env.common_step_counter,
      env.step_dt,
      original_stable_steps=self.ledger.settled_steps,
      active=rough,
      failed=self.failed,
      valid=valid & bounds & contact_valid,
    )
    self.completion_value.prepare(
      env.common_step_counter,
      env.step_dt,
      active=rough,
      finished=self.finished,
      failed=self.failed,
      valid=valid & bounds & contact_valid,
    )
    if self.stair_catalog is not None:
      hits = self.stair_catalog.contact_bits(
        env.sim.data,
        direction,
        self.stair_ledger.max_stage + 1,
        robot.data.root_link_pos_w[:, 0],
        rough & valid & ~self.failed,
      )
      self.stair_ledger.prepare(
        env.common_step_counter,
        env.step_dt,
        progress=progress,
        root=robot.data.root_link_pos_w,
        hits=hits,
        stages=self.stair_counts[levels],
        entry=self.ledger.next_gate >= 1,
        active=rough,
        failed=self.failed,
        valid=valid & bounds & contact_valid,
        birth_step=self.command.route_sample_step,
        direction=direction,
        level=levels,
      )
    fraction = ((foot_progress.amin(-1) - 0.75) / (length - 1.5)).clamp(0, 1)
    safe_observed = (
      (self.ledger.next_gate == 2)
      & safe_full_exit
      & (contact.sum(-1) >= 2)
      & valid
      & ~self.failed
    )
    flags = torch.stack(
      (
        self.ledger.next_gate >= 1,
        fraction >= 0.25,
        fraction >= 0.5,
        fraction >= 0.75,
        safe_observed,
        self.finished,
      ),
      -1,
    )
    flags &= (rough & valid & bounds & contact_valid & ~self.failed)[:, None]
    extra = torch.stack(
      (
        ~(valid & contact_valid),
        ~bounds,
        (progress - self.ledger.previous).abs() > p["maximum_step_progress"],
        ~torch.isfinite(robot.data.root_link_pos_w).all(-1),
      ),
      -1,
    )
    self.progress_credit.prepare(
      env.common_step_counter,
      env.step_dt,
      progress=progress,
      command_xy=self.command.command[:, :2],
      rough=rough,
      safe=valid & bounds & contact_valid & ~self.failed,
      supported=contact.sum(-1) >= 2,
      corridor=(footy.abs() <= (width - 0.02)[:, None]).all(-1),
      levels=levels,
      augmented=(
        self.routes.enabled if self.routes is not None else torch.zeros_like(rough)
      ),
    )
    self.episode_metrics.update(flags, self.ledger.settled_steps, contact, extra)
    passage_native.mark(
      self,
      root=robot.data.root_link_pos_w,
      levels=levels,
      direction=direction,
      rough=rough,
      safe=valid & bounds & contact_valid,
      exit_safe=safe_observed,
    )

  def record(self, ids):
    valid = (~self.command.flat_mask[ids]) & (self.env.episode_length_buf[ids] > 0)
    ids = ids[valid]
    if not len(ids):
      return
    passage_native.close(self, ids)
    grid = getattr(self, "grid", None)
    if grid is not None:
      terrain = require_terrain(self.env)
      rows = (
        torch.stack(
          (
            ids,
            self.command.route_sample_step[ids],
            terrain.terrain_levels[ids],
            torch.round(self.command.entry_angle[ids] * 180 / math.pi).long(),
            self.command.route_direction[ids],
            (self.finished[ids] & ~self.failed[ids]).long(),
          ),
          -1,
        )
        .cpu()
        .tolist()
      )
      records = [(i, b, h, a, d, bool(success)) for i, b, h, a, d, success in rows]
      if self.routes is not None:
        self.routes.flush(ids)
      normal = []
      fresh = []
      for row in records:
        i, b, *_ = row
        route = None if self.routes is None else self.routes.core.routes.get(i)
        if route is None:
          previous = grid.last_birth.get(i, -1)
        else:
          assert self.routes is not None
          previous = self.routes.core.last_birth.get(i, -1)
        if b > previous:
          fresh.append(i)
        if route is None:
          normal.append(row)
        else:
          assert self.routes is not None
          self.routes.core.finish(i, b, row[-1])
      grid.record_many(normal)
      if fresh:
        self.episode_metrics.finish(torch.tensor(fresh, device=self.env.device), self)
      self.history = grid.history
      return
    self.episode_metrics.finish(ids, self)
    terrain = self.env.scene.terrain
    # One batched host transfer on episode reset, never per-env GPU sync.
    rows = (
      torch.stack(
        (
          terrain.terrain_levels[ids],
          torch.round(self.command.entry_angle[ids] * 180 / math.pi).long(),
          self.command.route_direction[ids],
          (self.finished[ids] & ~self.failed[ids]).long(),
        ),
        -1,
      )
      .cpu()
      .tolist()
    )
    for level, angle, direction, success in rows:
      key = f"{level}/{angle}/{direction}"
      values = self.history.setdefault(key, [])
      values.append(bool(success))
      del values[: -self.parameters["window"]]
    qualified = True
    for angle in ENTRY_STAGES[self.angle_stage]:
      for direction in (-1, 1):
        values = self.history.get(f"{self.height_level}/{angle}/{direction}", [])
        qualified &= (
          len(values) >= self.parameters["minimum_trials"]
          and sum(values) / max(1, len(values)) >= self.parameters["success_fraction"]
        )
    if qualified:
      before = [self.height_level, self.angle_stage]
      if self.angle_stage + 1 < len(ENTRY_STAGES):
        self.angle_stage += 1
      elif self.height_level + 1 < len(self.cases):
        self.height_level += 1
        self.angle_stage = 0
      if before != [self.height_level, self.angle_stage]:
        self.promotions.append(
          dict(
            step=self.env.common_step_counter,
            before=before,
            after=[self.height_level, self.angle_stage],
          )
        )

  def state_dict(self):
    command = self.command
    record = dict(
      version=1,
      stair_progress=self.stair_ledger.state_dict(),
      exit_stability=self.exit_stability.state_dict(),
      completion_value=self.completion_value.state_dict(),
      episode_metrics=self.episode_metrics.state_dict(),
      progress_credit=self.progress_credit.state_dict(),
      task=command.cfg.task,
      parameters=self.parameters,
      height_level=self.height_level,
      angle_stage=self.angle_stage,
      history=copy.deepcopy(self.history),
      promotions=copy.deepcopy(self.promotions),
      ledger=self.ledger.state_dict(),
      last_step=self.last_step,
      tensors={
        name: getattr(self, name).detach().cpu().clone()
        for name in ("failed", "finished")
      },
      command={
        name: getattr(command, name).detach().cpu().clone()
        for name in ("route_direction", "entry_angle", "route_sample_step")
      },
    )

    if self.grid is not None:
      record["independent_grid"] = self.grid.state_dict()
      assert self.routes is not None
      record["safe_routes"] = self.routes.state_dict()
      assert self.flat_coverage is not None
      record["flat_coverage"] = self.flat_coverage.state_dict()
    passage_native.add_state(self, record)
    return record

  def load_state_dict(self, saved):
    passage_native.pre_restore(self, saved)
    if (
      saved["version"] != 1
      or saved["task"] != self.command.cfg.task
      or saved["parameters"] != self.parameters
    ):
      raise ValueError("Specialty resume task/config mismatch")
    self.height_level = saved["height_level"]
    self.angle_stage = saved["angle_stage"]
    if not 0 <= self.height_level < len(self.cases) or not 0 <= self.angle_stage < len(
      ENTRY_STAGES
    ):
      raise ValueError("Invalid curriculum level")
    self.history = copy.deepcopy(saved["history"])
    self.promotions = copy.deepcopy(saved["promotions"])
    self.ledger.load_state_dict(saved["ledger"])
    self.stair_ledger.load_state_dict(saved.get("stair_progress"))
    self.exit_stability.load_state_dict(saved.get("exit_stability"))
    self.completion_value.load_state_dict(saved.get("completion_value"))
    self.episode_metrics.load_state_dict(saved.get("episode_metrics"))
    self.progress_credit.load_state_dict(saved.get("progress_credit"))
    self.last_step = saved["last_step"]
    for target, values in (
      (self, saved["tensors"]),
      (self.command, saved["command"]),
    ):
      for name, value in values.items():
        current = getattr(target, name)
        if value.shape != current.shape or value.dtype != current.dtype:
          raise ValueError("Resume domain shape mismatch")
        current.copy_(value.to(self.env.device))

    if self.grid is not None:
      if "independent_grid" in saved:
        self.grid = GridCurriculum.from_state(saved["independent_grid"])
        if (
          self.grid.history != self.history
          or self.grid.highest_sampled != self.height_level
        ):
          raise ValueError(
            "Grid and legacy history/height compatibility fields disagree"
          )
      else:
        self.grid = GridCurriculum.from_legacy(
          self.history, self.height_level, self.angle_stage
        )
      self.history = self.grid.history
      self.flat_coverage = (
        FlatCoverage.from_state(saved["flat_coverage"])
        if "flat_coverage" in saved
        else FlatCoverage()
      )
      if self.routes is not None:
        if "safe_routes" in saved:
          self.routes.load_state_dict(saved["safe_routes"])
        else:
          self.routes.core = SafeRoutes()
          self.routes.sync(torch.arange(self.env.num_envs, device=self.env.device))
    passage_native.after_restore(self)


def specialty_state(env):
  if not hasattr(env, "specialty446_state"):
    env.specialty446_state = SpecialtyState(env)
  return env.specialty446_state


class SpecialtyEventTerm:
  def __init__(self, cfg, env):
    assert cfg.weight == 1
    self.state = specialty_state(env)

  def __call__(self, env):
    return self.state.ledger.consume(env.common_step_counter)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    if self.state.command.cfg.task != "stairs":
      selected = torch.arange(self.state.env.num_envs, device=self.state.env.device)[
        ids
      ]
      self.state.episode_metrics.reset(selected, self.state)
    self.state.ledger.reset(ids)


class StairProgressReward:
  def __init__(self, cfg, env):
    assert cfg.weight == 1
    self.state = specialty_state(env)

  def __call__(self, env):
    return self.state.stair_ledger.consume(env.common_step_counter)

  def reset(self, env_ids=None):
    ids = (
      torch.arange(self.state.env.num_envs, device=self.state.env.device)
      if env_ids is None or isinstance(env_ids, slice)
      else env_ids
    )
    self.state.episode_metrics.reset(ids, self.state)
    self.state.stair_ledger.reset(ids)


class ExitStabilityReward:
  def __init__(self, cfg, env):
    assert cfg.weight == 1
    self.state = specialty_state(env)

  def __call__(self, env):
    return self.state.exit_stability.consume(env.common_step_counter)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.state.exit_stability.reset(ids)


class CompletionValueReward:
  def __init__(self, cfg, env):
    assert cfg.weight == 1
    self.state = specialty_state(env)

  def __call__(self, env):
    return self.state.completion_value.consume(env.common_step_counter)

  def reset(self, env_ids=None):
    self.state.completion_value.reset(slice(None) if env_ids is None else env_ids)


def failure(env):
  st = specialty_state(env)
  st.update()
  return st.failed


def finish(env):
  st = specialty_state(env)
  st.update()
  return st.finished & ~st.failed


def assign_grid_routes(st, ids):
  if not len(ids):
    return
  if st.grid is None:
    raise ValueError("Independent grid only applies to stairs")
  if passage_native.assign(st, ids):
    return
  pending = [
    i
    for i in ids.cpu().tolist()
    if st.grid.assigned_birth.get(i) != st.env.common_step_counter
  ]
  if not pending:
    return
  ids = torch.tensor(pending, device=st.env.device, dtype=torch.long)
  if st.routes is None:
    rows = st.grid.sample(len(ids))
  else:
    assigned = st.routes.core.assign(
      pending,
      st.grid,
      st.lengths.cpu().tolist(),
      st.widths.cpu().tolist(),
      st.env.common_step_counter,
    )
    ids = torch.tensor(
      [row[0] for row in assigned], device=st.env.device, dtype=torch.long
    )
    rows = [row[1:] for row in assigned]
    st.routes.sync(ids)
  values = torch.tensor(rows, device=st.env.device, dtype=torch.long)
  terrain = require_terrain(st.env)
  terrain.terrain_levels[ids] = values[:, 0]
  st.command.entry_angle[ids] = values[:, 1] * math.pi / 180
  st.command.route_direction[ids] = values[:, 2]
  st.command.route_sample_step[ids] = st.env.common_step_counter
  st.grid.assigned_birth.update({i: st.env.common_step_counter for i in pending})
  terrain.env_origins[ids] = terrain.terrain_origins[
    terrain.terrain_levels[ids], terrain.terrain_types[ids]
  ]
  maximum = st.grid.highest_sampled
  if maximum > st.height_level:
    if maximum != st.height_level + 1:
      raise ValueError("Grid tried to skip a height")
    st.promotions.append(
      dict(
        step=st.env.common_step_counter,
        before=[st.height_level, st.angle_stage],
        after=[maximum, 0],
        kind="independent_grid_exploration_not_full_angle_mastery",
      )
    )
    st.height_level = maximum
    st.angle_stage = 0


def course_update(env, env_ids):
  st = specialty_state(env)
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  st.record(ids)
  terrain = env.scene.terrain
  rough = ids[~st.command.flat_mask[ids]]
  if st.grid is not None:
    assign_grid_routes(st, rough)
    return dict(
      height_level=st.height_level,
      independent_grid_version=GRID_POLICY["version"],
      passage_height_level=st.passage_course.current
      if st.passage_course is not None
      else -1,
      grid_currently_mastered=sum(
        v["currently_mastered"] for v in st.grid.summary()["cells"].values()
      ),
    )
  level = torch.full(
    (len(rough),), st.height_level, device=env.device, dtype=torch.long
  )
  # 20% retain previous learned heights; nobody skips the evidence ceiling.
  if st.height_level:
    replay = torch.rand(len(rough), device=env.device) < 0.2
    level[replay] = torch.randint(
      st.height_level, (int(replay.sum()),), device=env.device
    )
  terrain.terrain_levels[rough] = level
  terrain.env_origins[ids] = terrain.terrain_origins[
    terrain.terrain_levels[ids], terrain.terrain_types[ids]
  ]
  return dict(height_level=st.height_level, entry_angle_stage=st.angle_stage)


def reset_specialty(env, env_ids, pose_range, velocity_range):
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  command = env.command_manager.get_term("twist")
  flat = ids[command.flat_mask[ids]]
  if len(flat):
    common.reset_root_state_uniform(env, flat, pose_range, velocity_range)
  st = specialty_state(env)
  command.sample_route_entries(ids)
  rough = ids[~command.flat_mask[ids]]
  # Grouping bounds reset calls by height/direction/angle, not by env count.
  metadata = torch.stack(
    (
      env.scene.terrain.terrain_levels[rough],
      command.route_direction[rough],
      torch.round(command.entry_angle[rough] * 180 / math.pi).long(),
    ),
    -1,
  )
  for level, direction, angle in metadata.unique(dim=0).cpu().tolist():
    mask = (metadata == torch.tensor([level, direction, angle], device=env.device)).all(
      -1
    )
    ranges = copy.deepcopy(pose_range)
    offset = 0.0 if direction > 0 else float(st.lengths[level])
    yaw = (0.0 if direction > 0 else math.pi) + angle * math.pi / 180
    ranges.update(x=(offset, offset), y=(0.0, 0.0), yaw=(yaw, yaw))
    common.reset_root_state_uniform(env, rough[mask], ranges, velocity_range)
  if st.routes is not None:
    st.routes.spawn(rough)
  original_mdp.state(env).reset(ids)
  st.failed[ids] = False
  st.finished[ids] = False
  st.last_step = -1


def push_with_progress_grace(env, env_ids, velocity_range, asset_cfg=None):
  if asset_cfg is None:
    common.push_by_setting_velocity(env, env_ids, velocity_range)
  else:
    common.push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  specialty_state(env).progress_credit.on_external_push(ids)


def make_specialty_configs(
  task, *, course_parameters=None, num_envs=4096, seed=127001, passage_policy=None
):
  if task not in GROUPS:
    raise ValueError("Unknown specialty task")
  if passage_policy is not None and task != "stairs":
    raise ValueError("Passage course only applies to stairs")
  cfg, rl = comprehensive_configs(num_envs, seed, 112001, 124001)
  rows = len(GROUPS[task])
  terrains = {"flat": DedicatedFlatCfg(proportion=0.25)}
  for column, weight in enumerate((461, 461, 461, 461, 614, 614), start=1):
    terrains[f"{task}_route_{column}"] = SpecialtyTerrainCfg(
      task=task, column=column, proportion=weight / 4096
    )
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    seed=112001,
    curriculum=True,
    size=(80.0, 8.0),
    num_rows=rows,
    num_cols=7,
    border_width=4,
    sub_terrains=terrains,
    color_scheme="none",
  )
  if task in ("broken_bridge", "plum_stakes"):
    from support.black.sensors import GapAwareFootSensorCfg

    cfg.scene.sensors = tuple(
      GapAwareFootSensorCfg(
        **{
          field.name: copy.deepcopy(getattr(sensor, field.name))
          for field in fields(sensor)
        }
      )
      if sensor.name == "foot_height_scan"
      else sensor
      for sensor in cfg.scene.sensors
    )
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.sim.nconmax = 64
  old = cfg.commands["twist"]
  cfg.commands["twist"] = SpecialtyCommandCfg(
    **{
      f.name: copy.deepcopy(getattr(old, f.name))
      for f in fields(old)
      if f.name != "height_rows"
    },
    task=task,
    course_parameters=course_parameters,
    passage_course_policy=copy.deepcopy(passage_policy),
    height_rows=rows,
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["terrain_height_and_mask"].func = terrain_features
  # 仅坡道拟合地形法向；其余五项顶面均为水平面，对齐世界竖直方向避免垂直墙/空隙法向突变。
  if task != "slope":
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  # 越障专项只惩罚抬脚不足，不压制跨越20..40cm高台/高墙/台阶的高抬腿。
  cfg.rewards["foot_clearance"].params["penalize_overshoot"] = False
  cfg.rewards["foot_swing_height"].params["penalize_overshoot"] = False
  cfg.rewards["thigh_collision"] = RewardTermCfg(
    func=original_mdp.rough_collision,
    weight=-0.2,
    params={"sensor_name": "thigh_ground_touch"},
  )
  cfg.rewards["specialty_passage"] = RewardTermCfg(func=SpecialtyEventTerm, weight=1.0)
  cfg.terminations["failure"] = TerminationTermCfg(func=failure)
  if task == "stairs":
    cfg.rewards["specialty_stair_progress"] = RewardTermCfg(
      func=StairProgressReward, weight=1.0
    )
  # 全部六项专项均启用稳停塑形、吸收态尾值补偿(+12.0)与停滞扣分，消除非楼梯五专项的吸收态陷阱。
  cfg.rewards["specialty_exit_stability"] = RewardTermCfg(
    func=ExitStabilityReward, weight=1.0
  )
  verify_return_bound(cfg, rl)
  cfg.rewards["specialty_completion_value"] = RewardTermCfg(
    func=CompletionValueReward, weight=1.0
  )
  cfg.rewards["specialty_progress_dense_cost"] = RewardTermCfg(
    func=ProgressDenseCost, weight=-1.0
  )
  if cfg.events["push_robot"].func is not common.push_by_setting_velocity:
    raise ValueError("Do not replace an unknown external push implementation")
  cfg.events["push_robot"].func = push_with_progress_grace
  cfg.terminations["patch_traversal"] = TerminationTermCfg(func=finish, time_out=False)
  cfg.terminations.pop("out_of_map", None)
  cfg.curriculum = {"specialty_course": CurriculumTermCfg(func=course_update)}
  cfg.events["reset_base"].func = reset_specialty
  cfg.episode_length_s = 120.0
  configure_ppo_schedule(rl, "specialty")
  rl.experiment_name = "black_specialty_" + task
  return cfg, rl
