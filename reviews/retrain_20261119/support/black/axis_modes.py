"""Pure-axis mode exposure from original sampled targets and real observations."""

import copy

import torch

from support.black.context_coverage import CONTEXT_FIELDS

POLICY = dict(
  version=1,
  family=3,
  ordinary_pure_buckets=[0, 2, 4],
  counter_key="profile/axis/sign/original_speed_band",
  even_event="native_ramp_from_actual_previous_command",
  odd_event="original_abrupt",
  targets_profiles_directions_and_resampling_times_unchanged=True,
  native_linear_acceleration=1.0,
  native_yaw_acceleration=0.5,
  per_axis_direction_unique_quota=4,
  replay_protected_rows_per_original_layer=8,
  age_split_seconds=1.0,
  current_samples=1024,
  history_samples=1024,
  lambda_d=4,
)
KEYS = {
  f"{p}/{a}/{s}/{b}"
  for p in range(3)
  for a in range(3)
  for s in (-1, 1)
  for b in range(3)
}


class AxisEvents:
  def __init__(self):
    self.counts = dict.fromkeys(sorted(KEYS), 0)
    self.events = []

  def apply(self, command, ids, before, step):
    target = command.target_command[ids]
    profiles = command.profile_id[ids]
    active = target.abs() > 0.04
    bucket = command.task_class[ids]
    eligible = (
      command.flat_mask[ids]
      & (command.cycle_phase[ids] == -1)
      & (active.sum(-1) == 1)
      & ((target.abs() <= 1e-6) | active).all(-1)
      & (
        ((profiles == 0) & (bucket == 0))
        | ((profiles == 1) & (bucket == 2))
        | ((profiles == 2) & (bucket == 4))
      )
    )
    local = eligible.nonzero().flatten()
    if not len(local):
      return
    chosen_ids = ids[local]
    if (chosen_ids >= 1024).any() or (chosen_ids < 0).any():
      raise ValueError("Pure-axis coverage is flat-only")
    goals = target[local].detach().cpu()
    axes = goals.abs().argmax(-1)
    bands = torch.bucketize(goals.abs().amax(-1), torch.tensor([0.15, 0.5]))
    profile_values = profiles[local].cpu().tolist()
    timers = command.time_left[chosen_ids].detach().cpu().tolist()
    old = before[local].detach().cpu().tolist()
    modes = []
    for j, world in enumerate(chosen_ids.cpu().tolist()):
      axis = int(axes[j])
      sign = 1 if float(goals[j, axis]) > 0 else -1
      key = f"{profile_values[j]}/{axis}/{sign}/{int(bands[j])}"
      ordinal = self.counts[key]
      self.counts[key] += 1
      abrupt = bool(ordinal % 2)
      modes.append(abrupt)
      self.events.append(
        dict(
          env_id=world,
          step=int(step),
          key=key,
          ordinal=ordinal,
          target=goals[j].tolist(),
          previous_command=old[j],
          abrupt=abrupt,
          profile=profile_values[j],
          task_class=int(profile_values[j] * 2),
          phase=-1,
          resampling_seconds=timers[j],
          target_or_timer_modified=False,
        )
      )
    command.abrupt[chosen_ids] = torch.tensor(modes, device=command.device)
    ramp = torch.tensor([not m for m in modes], device=command.device)
    command.vel_command_b[chosen_ids[ramp]] = before[local[ramp]]
    command.context_family[chosen_ids] = 3
    self.events = self.events[-128:]

  def state_dict(self):
    return copy.deepcopy(dict(policy=POLICY, counts=self.counts, events=self.events))

  @classmethod
  def from_state(cls, saved):
    obj = cls()
    if saved is None:
      return obj
    if saved["policy"] != POLICY or set(saved["counts"]) != KEYS:
      raise ValueError("Unknown pure-axis coverage policy/counter keys")
    if any(type(n) is not int or n < 0 for n in saved["counts"].values()):
      raise ValueError("Invalid pure-axis event counters")
    if len(saved["events"]) > min(128, sum(saved["counts"].values())):
      raise ValueError("Invalid bounded pure-axis event history")
    for row in saved["events"]:
      if (
        type(row["env_id"]) is not int
        or not 0 <= row["env_id"] < 1024
        or row["phase"] != -1
        or type(row["profile"]) is not int
        or row["profile"] not in (0, 1, 2)
      ):
        raise ValueError("Nonflat or fabricated pure-axis phase")
      if (
        row["key"] not in KEYS
        or type(row["ordinal"]) is not int
        or not 0 <= row["ordinal"] < saved["counts"][row["key"]]
      ):
        raise ValueError("Invalid pure-axis event identity")
      if (
        type(row["step"]) is not int
        or row["step"] < 0
        or type(row["abrupt"]) is not bool
        or row["abrupt"] != bool(row["ordinal"] % 2)
      ):
        raise ValueError("Invalid pure-axis clock/mode")
      if (
        row["target_or_timer_modified"] is not False
        or not 0 < row["resampling_seconds"] <= 120
      ):
        raise ValueError("Pure-axis coverage cannot alter its original timer")
      goal = torch.tensor(row["target"])
      previous = torch.tensor(row["previous_command"])
      if (
        goal.shape != (3,)
        or previous.shape != (3,)
        or not torch.isfinite(goal).all()
        or not torch.isfinite(previous).all()
        or (goal.abs() > 0.04).sum() != 1
      ):
        raise ValueError("Invalid actual pure-axis command provenance")
      axis = int(goal.abs().argmax())
      sign = 1 if float(goal[axis]) > 0 else -1
      band = int(torch.bucketize(goal.abs().max(), torch.tensor([0.15, 0.5])))
      expected = f"{row['profile']}/{axis}/{sign}/{band}"
      if row["key"] != expected or row["task_class"] != row["profile"] * 2:
        raise ValueError("Pure-axis identity disagrees with actual sampled target")
    obj.counts = copy.deepcopy(saved["counts"])
    obj.events = copy.deepcopy(saved["events"])
    return obj


def axis_masks(command, phase, context):
  result = {
    f"axis{a}/{s}": torch.zeros(len(command), dtype=torch.bool)
    for a in range(3)
    for s in (-1, 1)
  }
  present = CONTEXT_FIELDS & set(context)
  if not present or phase is None:
    return result
  if present != CONTEXT_FIELDS:
    raise ValueError("Partial pure-axis provenance")
  target = context["event_target"]
  known = (
    (context["event_id"] >= 0)
    & (context["event_age"] >= 0)
    & (context["context_family"] == 3)
    & (phase == -1)
    & ((context["event_abrupt"] == 0) | (context["event_abrupt"] == 1))
  )
  pure = (target.abs() > 0.04).sum(-1) == 1
  for a in range(3):
    for s in (-1, 1):
      result[f"axis{a}/{s}"] = known & pure & (target[:, a] * s > 0.04)
  return result


def natural_reverse(context):
  previous = context["previous_command"]
  target = context["event_target"]
  return (
    ((previous.abs() > 0.04).sum(-1) == 1)
    & ((previous * target).sum(-1) < 0)
    & ((previous.abs().argmax(-1)) == target.abs().argmax(-1))
  )


def select_axis(ids, count, context, generator):
  if count == 0:
    return ids[:0]
  ordered = (
    ids if generator is None else ids[torch.randperm(len(ids), generator=generator)]
  )
  chosen = []
  seen = set()
  reverse = natural_reverse(context)

  def take(pool, unique):
    # Actual same-axis sign reversals get priority within the same mode/age cell.
    pool = torch.cat((pool[reverse[pool]], pool[~reverse[pool]]))
    for i in pool.tolist():
      key = (int(context["event_id"][i]), int(context["event_age"][i] >= 1.0))
      if len(chosen) >= count:
        return
      if i in chosen or (unique and key in seen):
        continue
      chosen.append(i)
      seen.add(key)
      return

  for mode in (0, 1):
    for early in (True, False):
      mask = (context["event_abrupt"][ordered] == mode) & (
        (context["event_age"][ordered] < 1.0) == early
      )
      take(ordered[mask], True)
  for unique in (True, False):
    while len(chosen) < min(count, len(ids)):
      before = len(chosen)
      take(ordered, unique)
      if len(chosen) == before:
        break
  if len(chosen) != min(count, len(ids)):
    raise ValueError("Cannot invent unavailable pure-axis states")
  return torch.tensor(chosen, dtype=torch.long)


def axis_metrics(current, ids, replay, mse):
  command = torch.cat((current["command"][ids], replay["command"]))
  phase = torch.cat((current["phase"][ids], replay["phase"]))
  context = {k: torch.cat((current[k][ids], replay[k])) for k in CONTEXT_FIELDS}
  reverse = natural_reverse(context)
  result = {}
  for name, mask in axis_masks(command, phase, context).items():
    result[name] = dict(
      rows=int(mask.sum()),
      ramp_rows=int((mask & (context["event_abrupt"] == 0)).sum()),
      abrupt_rows=int((mask & (context["event_abrupt"] == 1)).sum()),
      early_rows=int((mask & (context["event_age"] < 1.0)).sum()),
      natural_same_axis_reverse_rows=int((mask & reverse).sum()),
      actual_command_differs_from_target_rows=int(
        (mask & ((command - context["event_target"]).abs().amax(-1) > 1e-6)).sum()
      ),
      MSE=float(mse[mask].mean()) if mask.any() else None,
    )
  return result
