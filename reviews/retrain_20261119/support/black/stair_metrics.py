"""Episode denominators and terminal evidence, separate from course windows."""

import hashlib
import json
import math
from pathlib import Path

import torch

STAGES = (
  "entry",
  "quarter",
  "half",
  "three_quarters",
  "safe_full_exit",
  "strict_finish",
)


def terminal_details(env, state, ids):
  """One batched transfer at real episode end, before reset changes any state."""
  if not hasattr(state, "progress_credit"):
    return {}
  robot = env.scene["robot"]
  origin = env.scene.env_origins
  local = robot.data.root_link_pos_w - origin
  length = state.lengths[env.scene.terrain.terrain_levels]
  direction = state.command.route_direction
  progress = torch.where(direction > 0, local[:, 0], length - local[:, 0])
  feet = robot.data.site_pos_w[:, state.feet] - origin[:, None, :]
  fp = torch.where(
    direction[:, None] > 0, feet[:, :, 0], length[:, None] - feet[:, :, 0]
  )
  account = state.progress_credit
  guards = (
    state.routes.guard_bits
    if state.routes is not None
    else torch.zeros(env.num_envs, 4, device=env.device)
  )
  names = env.reward_manager.active_terms
  positive_names = (
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "pose",
  )
  values = dict(
    root_local=local,
    terrain_origin=origin,
    root_velocity_world=robot.data.root_link_lin_vel_w,
    actual_command=state.command.command,
    foot_progress=fp,
    foot_local_z=feet[:, :, 2],
    progress=progress[:, None],
    dense_gate=account.gate[:, None],
    window_credible_progress=account.window_credit[:, None],
    window_demand=account.window_demand[:, None],
    episode_age=account.age[:, None],
    recovery_left=account.recovery_left[:, None],
    raw_progress_high_water=account.raw_best[:, None],
    credited_progress=account.credit[:, None],
    positive_withheld_rate=account.withheld_rate[:, None],
    route_guard_bits=guards.float(),
    public_positive_rates=env.reward_manager._step_reward[
      :, [names.index(k) for k in positive_names]
    ],
  )
  values.update(state.control_snapshot)
  widths = {k: v.shape[1] for k, v in values.items()}
  packed = torch.cat(tuple(values.values()), -1)[ids].cpu().tolist()
  result = {}
  for world, row in zip(ids.cpu().tolist(), packed, strict=True):
    sample = {}
    offset = 0
    for key, width in widths.items():
      section = [v if math.isfinite(v) else None for v in row[offset : offset + width]]
      sample[key] = section[0] if width == 1 else section
      offset += width
    sample["route_guard_reason_bits"] = dict(
      zip(
        ("footprint", "cross_track", "heading", "nonfinite"),
        [bool(v) for v in sample.pop("route_guard_bits")],
        strict=True,
      )
    )
    sample["public_positive_terms"] = list(positive_names)
    sample["snapshot_semantics"] = (
      "Actual terminal task/reward snapshot before physical reset; not a full trajectory"
    )
    result[world] = sample
  return result


class EpisodeMetrics:
  fields = (
    "serial",
    "active",
    "seen",
    "first",
    "birth",
    "birth_policy",
    "max_stable",
    "exit_samples",
    "exit_missing_four",
    "extra_reasons",
  )

  def __init__(self, env):
    self.env = env
    n = env.num_envs
    self.serial = torch.zeros(n, device=env.device, dtype=torch.long)
    self.active = torch.zeros(n, device=env.device, dtype=torch.bool)
    self.seen = torch.zeros(n, len(STAGES), device=env.device, dtype=torch.bool)
    self.first = torch.full((n, len(STAGES)), -1, device=env.device, dtype=torch.long)
    self.birth = torch.zeros_like(self.serial)
    self.birth_policy = torch.zeros_like(self.serial)
    self.max_stable = torch.zeros_like(self.serial)
    self.exit_samples = torch.zeros_like(self.serial)
    self.exit_missing_four = torch.zeros_like(self.serial)
    self.extra_reasons = torch.zeros(n, 4, device=env.device, dtype=torch.bool)
    self.path = None
    self.run_id = "unconfigured"
    self.context = {}
    self.policy_update = 0
    self.event_seq = 0
    self.last_step = -1

  def configure(self, output, context):
    self.path = Path(output) / "EPISODE_STAGE_EVIDENCE.jsonl"
    self.run_id = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    self.context = context
    self.policy_update = context["source_actual_updates"]

  def emit(self, rows):
    if self.path is None or not rows:
      return
    for row in rows:
      self.event_seq += 1
      row.update(
        event_seq=self.event_seq,
        context=self.context,
        policy_kind=self.context.get(
          "policy_kind", "changing_during_training_not_fixed_evaluation"
        ),
      )
    with self.path.open("a") as stream:
      stream.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))

  def update(self, flags, stable, contacts, extra_reasons):
    step = self.env.common_step_counter
    if self.last_step == step:
      return
    self.last_step = step
    self.extra_reasons |= extra_reasons
    fresh = flags & ~self.seen & self.active[:, None]
    self.first = torch.where(fresh, step, self.first)
    self.seen |= flags & self.active[:, None]
    self.max_stable = torch.maximum(self.max_stable, stable)
    exiting = self.active & self.seen[:, 4]
    self.exit_samples += exiting.long()
    self.exit_missing_four += (exiting & ~contacts.all(-1)).long()

  def finish(self, ids, state, *, censored=None):
    ids = ids[self.active[ids]]
    if not len(ids):
      return
    if self.path is None:
      self.active[ids] = False
      return
    env = self.env
    command = state.command
    baseline = env.black_rough_state
    timeouts = (
      torch.zeros_like(state.failed)
      if censored
      else getattr(env, "reset_time_outs", torch.zeros_like(state.failed))
    )
    rows = (
      torch.cat(
        (
          ids[:, None],
          self.serial[ids, None],
          self.birth[ids, None],
          self.birth_policy[ids, None],
          env.scene.terrain.terrain_levels[ids, None],
          torch.round(command.entry_angle[ids] * 180 / torch.pi).long()[:, None],
          command.route_direction[ids, None],
          self.seen[ids].long(),
          self.first[ids],
          self.max_stable[ids, None],
          self.exit_samples[ids, None],
          self.exit_missing_four[ids, None],
          state.failed[ids, None].long(),
          state.finished[ids, None].long(),
          timeouts[ids, None].long(),
          baseline.reason[ids].long(),
          state.ledger.paid_ticks[ids, None],
          state.stair_ledger.paid_milli[ids, None],
          self.extra_reasons[ids].long(),
          state.stair_ledger.max_stage[ids, None],
          (
            # 同批打包整数计数；浮点奖励换成微单位，误差最多0.5e-6。
            (state.exit_stability.paid_total[ids, None] * 1_000_000).round().long()
            if hasattr(state, "exit_stability")
            else torch.zeros_like(self.serial[ids, None])
          ),
          (
            state.completion_value.paid[ids, None].long()
            if hasattr(state, "completion_value")
            else torch.zeros_like(self.serial[ids, None])
          ),
        ),
        -1,
      )
      .cpu()
      .tolist()
    )
    result = []
    for values in rows:
      (world, serial, birth, policy, level, angle, direction) = values[:7]
      seen = [bool(v) for v in values[7:13]]
      first = values[13:19]
      maximum, exit_n, no4, failed, finished, timeout = values[19:25]
      reasons = values[25:29]
      exit_time = first[4]
      outcome = (
        "censored"
        if censored
        else "failure"
        if failed
        else "strict_success"
        if finished
        else "timeout"
        if timeout
        else "unknown_reset"
      )
      stage = next((STAGES[i] for i in range(5, -1, -1) if seen[i]), "before_entry")
      result.append(
        dict(
          kind="end",
          episode_uid=f"{self.run_id}:{world}:{serial}",
          env_id=world,
          birth_step=birth,
          terminal_step=env.common_step_counter,
          policy_update_start=policy,
          policy_update_end=self.policy_update,
          height_level=level,
          entry_angle_degrees=angle,
          direction=direction,
          stages=dict(zip(STAGES, seen, strict=True)),
          first_steps=dict(zip(STAGES, first, strict=True)),
          outcome=outcome,
          censor_reason=censored,
          terminal_stage=stage,
          failure_reason_bits=dict(
            zip(("thigh", "trunk", "tilt", "clearance"), reasons, strict=True)
          ),
          failure_without_baseline_bits=bool(failed and not any(reasons)),
          raw_timeout=None if censored else bool(timeout),
          reason_semantics="Observed safety flags; not exclusive causal attribution",
          raw_strict_finished=bool(finished),
          max_stable_control_steps=maximum,
          exit_control_samples=exit_n,
          exit_missing_four_samples=no4,
          exit_wait_seconds=(
            None
            if exit_time < 0
            else (env.common_step_counter - exit_time) * env.step_dt
          ),
          exit_wait_completed=outcome == "strict_success",
          original_event_units=values[29] / 4,
          ascent_event_units=values[30] / 1000,
          max_legal_ascent_stage=values[35],
          exit_stability_event_units=values[36] / 1_000_000,
          completion_value_event_units=values[37] * 12.0,
          extra_failure_reason_bits=dict(
            zip(
              ("invalid_sensor", "route_bounds", "discontinuity", "nonfinite_root"),
              values[31:35],
              strict=True,
            )
          ),
        )
      )
    if getattr(state, "routes", None) is not None:
      state.routes.flush(ids)
      for row in result:
        row["entry_angle_semantics"] = "nominal_grid_birth_bin_not_measured_contact"
        row["nominal_entry_angle_degrees"] = row["entry_angle_degrees"]
        route = state.routes.core.routes.get(row["env_id"])
        row["actual_path_evidence"] = route
        row["route_kind"] = "legacy_grid" if route is None else route["kind"]
    details = {} if censored else terminal_details(env, state, ids)
    for row in result:
      if row["env_id"] in details:
        row["terminal_motion_and_reward"] = details[row["env_id"]]
    self.emit(result)
    self.active[ids] = False

  def reset(self, ids, state):
    # An external reset is explicit censoring, never an invented failure.
    self.finish(ids, state, censored="external_reset_or_process_restart")
    env = self.env
    rough = ~state.command.flat_mask[ids]
    self.serial[ids] += 1
    self.active[ids] = rough
    self.seen[ids] = False
    self.first[ids] = -1
    self.extra_reasons[ids] = False
    self.birth[ids] = env.common_step_counter
    self.birth_policy[ids] = self.policy_update
    for key in ("max_stable", "exit_samples", "exit_missing_four"):
      getattr(self, key)[ids] = 0
    selected = ids[rough]
    values = (
      torch.stack(
        (
          selected,
          self.serial[selected],
          env.scene.terrain.terrain_levels[selected],
          torch.round(state.command.entry_angle[selected] * 180 / torch.pi).long(),
          state.command.route_direction[selected],
        ),
        -1,
      )
      .cpu()
      .tolist()
    )
    self.emit(
      [
        dict(
          kind="start",
          episode_uid=f"{self.run_id}:{i}:{serial}",
          env_id=i,
          birth_step=env.common_step_counter,
          policy_update_start=self.policy_update,
          height_level=level,
          entry_angle_degrees=angle,
          entry_angle_semantics="nominal_grid_birth_bin_not_measured_contact",
          nominal_entry_angle_degrees=angle,
          actual_path_evidence=(
            None
            if getattr(state, "routes", None) is None
            else state.routes.core.routes.get(i)
          ),
          direction=direction,
        )
        for i, serial, level, angle, direction in values
      ]
    )

  def state_dict(self):
    return dict(
      version=1,
      run_id=self.run_id,
      event_seq=self.event_seq,
      last_step=self.last_step,
      policy_update=self.policy_update,
      context=self.context,
      tensors={k: getattr(self, k).detach().cpu().clone() for k in self.fields},
    )

  def load_state_dict(self, saved):
    if saved is None:
      self.active.zero_()
      return
    if saved["version"] != 1:
      raise ValueError("Unknown episode evidence schema")
    for key in self.fields:
      source = saved["tensors"][key]
      target = getattr(self, key)
      if source.shape != target.shape or source.dtype != target.dtype:
        raise ValueError("Invalid metrics state " + key)
      target.copy_(source.to(target.device))
    self.run_id = saved["run_id"]
    self.event_seq = saved["event_seq"]
    self.last_step = saved["last_step"]
    self.policy_update = saved["policy_update"]
    self.context = saved.get("context", {})
