"""Real flat transition contexts, retaining the original event-direction mix."""

import copy
import random
from typing import TypedDict

import torch

from support.black.grid_course import tuple_tree
from support.black.command_coverage import POLICY as POLICY
from support.black.command_coverage import FlatCoverage as LegacyCoverage
from support.black.command_coverage import target_masks as legacy_target_masks


class ContextPolicy(TypedDict):
  version: int
  training_seed: int
  low_target: list[float]
  low_neighbor_radius: float
  matching_low_mixed_event_fraction: float
  retention_target: list[float]
  retention_neighbor_radius: float
  existing_retention_augmented_fraction_to_cycle: float
  native_phase_seconds: list[float]
  low_inner_radius: float
  low_outer_radius: float
  target_draw_floor: int
  phase1_and_phase2_preferred_unique_quota: int
  flat_count: int
  current_samples: int
  history_samples: int
  lambda_d: int
  cache_capacity: int
  source: str


CONTEXT_POLICY: ContextPolicy = dict(
  version=1,
  training_seed=2026111327,
  low_target=[0.12, -0.23, 0.07],
  low_neighbor_radius=0.01,
  matching_low_mixed_event_fraction=0.5,
  retention_target=[0.65, 0.5, -0.4],
  retention_neighbor_radius=0.05,
  existing_retention_augmented_fraction_to_cycle=0.5,
  native_phase_seconds=[1.0, 5.0, 5.0, 3.0],
  low_inner_radius=0.015,
  low_outer_radius=0.03,
  target_draw_floor=8,
  phase1_and_phase2_preferred_unique_quota=2,
  flat_count=1024,
  current_samples=1024,
  history_samples=1024,
  lambda_d=4,
  cache_capacity=200000,
  source="actual flat events only, training RNG disjoint from heldout seeds",
)
TARGETS = {
  "retention22": ([0.65, 0.5, -0.4], 0.1, 0.05),
  "retention22_mirror": ([-0.65, -0.5, 0.4], 0.1, 0.05),
  "low12": ([0.12, -0.23, 0.07], 0.03, 0.015),
  "low12_mirror": ([-0.12, 0.23, -0.07], 0.03, 0.015),
}
CONTEXT_FIELDS = {
  "previous_command",
  "event_target",
  "event_age",
  "event_id",
  "context_family",
  "event_abrupt",
}


def target_masks(command, phase):
  result = legacy_target_masks(command, phase)
  known = (
    torch.zeros(len(command), dtype=torch.bool, device=command.device)
    if phase is None
    else ((phase == -1) | (phase == 1) | (phase == 2))
  )
  for name, (goal, radius, _) in TARGETS.items():
    if name in result:
      continue
    goal = torch.tensor(goal, dtype=command.dtype, device=command.device)
    result[name] = (
      known
      & ((command - goal).abs().amax(-1) <= radius + 1e-7)
      & ((command * goal.sign()) > 0.04).all(-1)
    )
  return result


def initialize_context(command):
  """Unknown until an actual native command event is observed."""
  device = command.device
  n = command.num_envs
  command.context_previous = torch.zeros(n, 3, device=device)
  command.context_target = torch.zeros(n, 3, device=device)
  command.context_event_step = torch.full((n,), -1, dtype=torch.long, device=device)
  command.context_family = torch.full((n,), -1, dtype=torch.long, device=device)
  command.context_cycle = torch.zeros(n, dtype=torch.bool, device=device)


def reset_context(command, ids):
  if not hasattr(command, "context_previous"):
    return
  if ids is None:
    ids = slice(None)
  command.context_previous[ids] = 0
  command.context_target[ids] = 0
  command.context_event_step[ids] = -1
  command.context_family[ids] = -1
  command.context_cycle[ids] = False


def context_batch(command, ids, step, dt):
  event_step = command.context_event_step[ids]
  known = event_step >= 0
  age = torch.where(known, (int(step) - event_step).float() * dt, -1.0)
  if (age[known] < 0).any():
    raise ValueError("Future command event cannot label an observation")
  return dict(
    previous_command=command.context_previous[ids].detach().cpu().clone(),
    event_target=command.context_target[ids].detach().cpu().clone(),
    event_age=age.detach().cpu().clone(),
    event_id=torch.where(known, event_step * 4096 + ids, -1).detach().cpu().clone(),
    context_family=command.context_family[ids].detach().cpu().clone(),
    event_abrupt=torch.where(known, command.abrupt[ids].long(), -1)
    .detach()
    .cpu()
    .clone(),
  )


def unknown_context(n):
  return dict(
    previous_command=torch.zeros(n, 3),
    event_target=torch.zeros(n, 3),
    event_age=torch.full((n,), -1.0),
    event_id=torch.full((n,), -1, dtype=torch.long),
    context_family=torch.full((n,), -1, dtype=torch.long),
    event_abrupt=torch.full((n,), -1, dtype=torch.long),
  )


class FlatCoverage(LegacyCoverage):
  def __init__(self):
    super().__init__()
    self.context_rng = random.Random(CONTEXT_POLICY["training_seed"])
    self.context_counts = {
      "low_positive": 0,
      "low_mirror": 0,
      "ret_positive": 0,
      "ret_mirror": 0,
    }
    self.context_events = []
    self.context_total = 0

  def resample(self, command, ids, before, step):
    # Existing ret22 half-event augmentation and its RNG/counters stay intact.
    original_target = command.target_command[ids].clone()
    old_counts = dict(self.counts)
    super().resample(command, ids, before, step)
    if not hasattr(command, "context_previous"):
      initialize_context(command)
    flat = command.flat_mask[ids]
    local_ids = ids[flat].cpu().tolist()
    lookup = {int(i): j for j, i in enumerate(ids.cpu().tolist())}
    for i in local_ids:
      if int(command.cycle_phase[i]) == -1:
        command.context_cycle[i] = False
    selected = []
    for i in local_ids:
      goal = command.target_command[i]
      if (
        int(command.profile_id[i]) != 0
        or int(command.task_class[i]) != 1
        or int(command.cycle_phase[i]) != -1
      ):
        continue
      sign = (
        1
        if bool(
          (goal * torch.tensor([1.0, -1.0, 1.0], device=goal.device) > 0.04).all()
        )
        else -1
        if bool(
          (goal * torch.tensor([-1.0, 1.0, -1.0], device=goal.device) > 0.04).all()
        )
        else 0
      )
      if not sign:
        continue
      name = "low_positive" if sign > 0 else "low_mirror"
      ordinal = self.context_counts[name]
      self.context_counts[name] += 1
      if ordinal % 2:
        continue
      variant = (ordinal // 2) % 4
      target = [sign * v for v in CONTEXT_POLICY["low_target"]]
      if variant in (1, 3):
        target = [v + self.context_rng.uniform(-0.01, 0.01) for v in target]
      selected.append((i, target, variant in (0, 3), 1, name, ordinal))
    # The legacy telemetry retains only128 events. Reconstruct the complete
    # current call's selection from its original event mask and ordinals,
    # rather than using that truncated telemetry as a control input.
    rows = []
    for i in local_ids:
      j = lookup[i]
      original = original_target[j]
      if (
        int(command.profile_id[i]) != 1
        or int(command.task_class[i]) != 3
        or int(command.cycle_phase[i]) != -1
        or float(original.abs().max()) <= 0.5
      ):
        continue
      sign = (
        "positive"
        if bool(
          (
            original * torch.tensor([1.0, 1.0, -1.0], device=original.device) > 0.04
          ).all()
        )
        else "mirror"
        if bool(
          (
            original * torch.tensor([-1.0, -1.0, 1.0], device=original.device) > 0.04
          ).all()
        )
        else None
      )
      if sign is None:
        continue
      ordinal = old_counts[sign]
      old_counts[sign] += 1
      if ordinal % 2 == 0:
        rows.append(dict(env_id=i, sign=sign))
    for row in rows:
      i = row["env_id"]
      name = "ret_positive" if row["sign"] == "positive" else "ret_mirror"
      ordinal = self.context_counts[name]
      self.context_counts[name] += 1
      if ordinal % 2:
        continue
      sign = 1 if row["sign"] == "positive" else -1
      variant = (ordinal // 2) % 4
      target = [sign * v for v in CONTEXT_POLICY["retention_target"]]
      if variant in (1, 3):
        target = [v + self.context_rng.uniform(-0.05, 0.05) for v in target]
      selected.append((i, target, variant in (0, 3), 2, name, ordinal))
    for i, target, abrupt, family, name, ordinal in selected:
      if not bool(command.flat_mask[i]):
        raise ValueError("No rough context injection")
      command.task_class[i] = 5
      command.cycle_target[i] = torch.tensor(target, device=command.device)
      command.abrupt[i] = abrupt
      command.cycle_phase[i] = 0 if int(command._env.episode_length_buf[i]) == 0 else 1
      one = torch.tensor([i], device=command.device)
      command._set_phase(one)
      command.context_family[i] = family
      command.context_cycle[i] = True
      self.context_events.append(
        dict(
          env_id=i,
          step=int(step),
          family=name,
          ordinal=ordinal,
          previous_command=before[lookup[i]].cpu().tolist(),
          cycle_target=target,
          abrupt=abrupt,
          phase=int(command.cycle_phase[i]),
          profile=int(command.profile_id[i]),
        )
      )
      self.context_total += 1
    for i in local_ids:
      j = lookup[i]
      if not bool(command.context_cycle[i]):
        command.context_family[i] = 0
      command.context_previous[i] = before[j]
      command.context_target[i] = command.target_command[i]
      command.context_event_step[i] = int(step)
      if bool(command.abrupt[i]) or int(command.cycle_phase[i]) == 0:
        command.vel_command_b[i] = command.target_command[i]
      else:
        command.vel_command_b[i] = before[j]
    self.context_events = self.context_events[-128:]

  def state_dict(self):
    result = super().state_dict()
    result["transition_context_v27"] = copy.deepcopy(
      dict(
        policy=CONTEXT_POLICY,
        rng=self.context_rng.getstate(),
        counts=self.context_counts,
        total=self.context_total,
        events=self.context_events,
      )
    )
    return result

  @classmethod
  def from_state(cls, saved):
    legacy = dict(saved)
    state = legacy.pop("transition_context_v27", None)
    old = LegacyCoverage.from_state(legacy)
    obj = cls()
    obj.__dict__.update(old.__dict__)
    if state is None:
      return obj
    if state["policy"] != CONTEXT_POLICY:
      raise ValueError("Unknown transition context policy")
    if set(state["counts"]) != set(obj.context_counts) or any(
      type(v) is not int or v < 0 for v in state["counts"].values()
    ):
      raise ValueError("Invalid transition counters")
    if type(state["total"]) is not int or not 0 <= state["total"] <= sum(
      state["counts"].values()
    ):
      raise ValueError("Invalid transition totals")
    if len(state["events"]) > 128 or len(state["events"]) > state["total"]:
      raise ValueError("Invalid transition event ledger")
    for row in state["events"]:
      if (
        not 0 <= row["env_id"] < 1024
        or row["family"] not in obj.context_counts
        or row["phase"] not in (0, 1)
        or type(row["abrupt"]) is not bool
      ):
        raise ValueError("Invalid transition event identity")
      if (
        type(row["step"]) is not int
        or row["step"] < 0
        or type(row["ordinal"]) is not int
        or row["ordinal"] < 0
      ):
        raise ValueError("Invalid transition clock")
      base = (
        CONTEXT_POLICY["low_target"]
        if row["family"].startswith("low")
        else CONTEXT_POLICY["retention_target"]
      )
      radius = 0.01 if row["family"].startswith("low") else 0.05
      sign = -1 if row["family"].endswith("mirror") else 1
      if (
        len(row["cycle_target"]) != 3
        or max(
          abs(v - sign * b) for v, b in zip(row["cycle_target"], base, strict=True)
        )
        > radius + 1e-7
      ):
        raise ValueError("Unbounded transition target")
      if not torch.isfinite(
        torch.tensor(row["previous_command"] + row["cycle_target"])
      ).all():
        raise ValueError("Nonfinite transition metadata")
    obj.context_rng.setstate(tuple_tree(state["rng"]))
    obj.context_counts = copy.deepcopy(state["counts"])
    obj.context_events = copy.deepcopy(state["events"])
    obj.context_total = state["total"]
    return obj
