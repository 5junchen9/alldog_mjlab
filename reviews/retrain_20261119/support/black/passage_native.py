"""Course adapter for the exact v7_fix1 native stairs; legacy strict ledger retained."""

import math

import torch

from support.black.passage_course import (
  FROZEN,
  HEIGHTS,
  PassageCourse,
  schedule_classify,
)
from support.black.passage_tracks import ObservedPassage, PhysicalTrack


def init(st):
  st.passage_course = PassageCourse(st.command.cfg.passage_course_policy)
  st.passage_track = PhysicalTrack(st.env.sim.mj_model, st.stair_catalog, st.env.device)
  if not all(
    h in st.passage_track.compiled_cm
    for h in HEIGHTS
    if h <= st.passage_course.policy["maximum_cm"]
  ):
    raise ValueError("Compiled geometry is missing an eligible physical height")
  st.passage_seen = ObservedPassage(st.env.num_envs, st.env.device)
  st.passage_live_restore = False
  st.passage_bridge = "fresh_candidate_legacy_strict_histories_never_counted"
  assert st.command.cfg.task == "stairs"


def mark(st, *, root, levels, direction, rough, safe, exit_safe):
  if st.passage_course is None:
    return
  env = st.env
  ob = st.passage_seen
  # All flags come from the original terminal/reward snapshot, after final original failure.
  safe_exit = (
    exit_safe
    & (st.ledger.next_gate == 2)
    & st.ledger.crossed_entry
    & st.ledger.crossed_exit
    & ~st.ledger.invalid
  )
  hit, landing = st.passage_track.contacts(
    env.sim.data,
    direction,
    levels,
    ob.stage + 1,
    root[:, 0],
    rough & ob.assigned & safe & ~st.failed,
  )
  ob.observe(
    env.common_step_counter,
    root=root,
    birth=st.command.route_sample_step,
    level=levels,
    direction=direction,
    active=rough,
    bad=(st.failed | ~safe | st.ledger.invalid),
    entry=(st.ledger.next_gate >= 1) & st.ledger.crossed_entry,
    native_ascent=st.stair_ledger.max_stage,
    exit_ok=safe_exit,
    hit=hit,
    landing=landing,
  )


def close(st, ids):
  if st.passage_course is None:
    return
  book = st.passage_course
  state = st.passage_seen
  env = st.env
  channels = torch.stack(
    (
      ids,
      st.command.route_sample_step[ids],
      env.scene.terrain.terrain_levels[ids],
      torch.round(st.command.entry_angle[ids] * 180 / math.pi).long(),
      st.command.route_direction[ids],
      state.pass_once[ids].long(),
      st.failed[ids].long(),
      state.invalid[ids].long(),
      env.reset_time_outs[ids].long(),
      env.reset_terminated[ids].long(),
      st.finished[ids].long(),
      state.stage[ids],
      state.landing[ids].long(),
      state.first_verified_pass_tick[ids],
    ),
    -1,
  )
  for (
    i,
    b,
    h,
    angle,
    direction,
    won,
    bad,
    invalid,
    timeout,
    terminated,
    strict,
    steps,
    landing,
    first,
  ) in channels.cpu().tolist():
    bad = bool(bad or (terminated and not strict))
    # Administrative reset / extraction is neither success nor terminal failure.
    outcome = (
      schedule_classify(bool(won), bad, bool(timeout), bool(strict), bool(invalid))
      if (timeout or terminated)
      else "censored"
    )
    proof = dict(
      ordered_treads=steps,
      landing_contact=bool(landing),
      first_verified_pass_tick=first,
      closed_tick=env.common_step_counter,
    )
    if book.vote(i, b, h, angle, direction, outcome, proof):
      state.expire(i)
  book.advance_once(env.common_step_counter)


def assign(st, ids):
  book = st.passage_course
  if book is None:
    return False
  env = st.env
  t = env.scene.terrain
  stamp = env.common_step_counter
  pending = [i for i in ids.cpu().tolist() if st.grid.assigned_birth.get(i) != stamp]
  if not pending:
    return True
  index = torch.tensor(pending, device=env.device, dtype=torch.long)
  planned = [book.assign(i, stamp) for i in pending]
  # Auxiliary uses the original ordinary ±15 geometry/reset, not an extra independent
  # low-route 10% overlay; that would exceed declared aux and confuse head-on evidence.
  for i in pending:
    st.routes.core.routes.pop(i, None)
  st.routes.sync(index)
  values = torch.tensor(planned, device=env.device, dtype=torch.long)
  t.terrain_levels[index] = values[:, 0]
  st.command.entry_angle[index] = values[:, 1] * math.pi / 180
  st.command.route_direction[index] = values[:, 2]
  st.command.route_sample_step[index] = stamp
  st.grid.assigned_birth.update({i: stamp for i in pending})
  st.grid.retained.update(f"{h}/{angle}/{direction}" for h, angle, direction in planned)
  t.env_origins[index] = t.terrain_origins[values[:, 0], t.terrain_types[index]]
  # Old strict diagnostics can still grow with physical sample identity, never gate new passage.
  max_sample = int(values[:, 0].max())
  st.grid.highest_sampled = max(st.grid.highest_sampled, max_sample)
  st.height_level = max(st.height_level, max_sample)
  st.passage_seen.on_assign(index, stamp, values[:, 0], values[:, 2])
  return True


def add_state(st, record):
  if st.passage_course is not None:
    ob = st.passage_seen
    record["passage_course_candidate"] = dict(
      book=st.passage_course.state_dict(),
      bridge=st.passage_bridge,
      physical_inflight_not_restorable=True,
      inflight_censor_required=True,
      source_model_sha=FROZEN,
      compiled_cm=st.passage_track.compiled_cm,
      stage_snapshot={
        n: getattr(ob, n).detach().cpu().clone()
        for n in (
          "stage",
          "stage_contact_bits",
          "landing",
          "pass_once",
          "invalid",
          "birth",
          "first_verified_pass_tick",
        )
      },
    )


def pre_restore(st, saved):
  if st.passage_course is None:
    if "passage_course_candidate" in saved:
      raise ValueError("Passage checkpoint cannot hot-load into old course")
    return
  if "passage_course_candidate" not in saved:
    raise ValueError(
      "Old course checkpoint cannot hot-load into passage; use offline bridge"
    )
  bag = saved["passage_course_candidate"]
  assert (
    bag["physical_inflight_not_restorable"]
    and bag["inflight_censor_required"]
    and bag["source_model_sha"] == FROZEN
  )
  assert bag["compiled_cm"] == st.passage_track.compiled_cm
  if st.passage_course.active:
    raise ValueError(
      "Do not replace course state while native candidate episodes exist"
    )
  st.passage_course = PassageCourse.restore(
    bag["book"],
    st.command.cfg.passage_course_policy,
    cold_reset=True,
    explicit_candidate=True,
  )
  st.passage_bridge = bag["bridge"]
  st.passage_live_restore = True


def after_restore(st):
  if st.passage_course is not None and st.passage_live_restore:
    # Old assigned birth from checkpoint can equal current clock.
    # New physical reset must produce fresh native birth, never continue old lane votes.
    st.grid.assigned_birth.clear()
    st.passage_seen.assigned.zero_()
    # Permit the next actual reset tick only if genuinely newer than the censored saved token.
    if (
      st.passage_course.last_closed
      and max(st.passage_course.last_closed.values()) >= st.env.common_step_counter
    ):
      raise ValueError(
        "Cold candidate restore must bind a new native tick after old lane tokens"
      )
    st.passage_live_restore = False
