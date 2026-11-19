"""Bounded real command-neighborhood coverage, preserving other event directions."""

import copy
import math
import random
from typing import TypedDict

import torch

from support.black.grid_course import tuple_tree


class FlatPolicy(TypedDict):
  version: int
  target: list[float]
  neighbor_radius: float
  supervision_radius: float
  per_sign_target_fraction: float
  CPU_seed: int
  source: str
  event_sequence: list[str]


POLICY: FlatPolicy = dict(
  version=1,
  target=[0.65, 0.5, -0.4],
  neighbor_radius=0.05,
  supervision_radius=0.1,
  per_sign_target_fraction=0.5,
  CPU_seed=201126001,
  source="existing flat omni mixed ordinary events in same sign/high-speed cell",
  event_sequence=[
    "exact_abrupt",
    "original",
    "neighbor_ramp",
    "original",
    "exact_ramp",
    "original",
    "neighbor_abrupt",
    "original",
  ],
)


def target_masks(command, phase):
  if phase is None:
    return {
      name: torch.zeros(len(command), dtype=torch.bool)
      for name in ("retention22", "retention22_mirror")
    }
  known = (phase == -1) | (phase == 1) | (phase == 2)
  goal = torch.tensor(POLICY["target"], dtype=command.dtype, device=command.device)
  return dict(
    retention22=known & ((command - goal).abs().amax(-1) <= 0.1000001),
    retention22_mirror=known & ((command + goal).abs().amax(-1) <= 0.1000001),
  )


class FlatCoverage:
  def __init__(self):
    self.rng = random.Random(POLICY["CPU_seed"])
    self.counts = {"positive": 0, "mirror": 0}
    self.events = []
    self.total_augmented = 0

  def resample(self, command, ids, before, step):
    flat = command.flat_mask[ids]
    eligible = (
      flat
      & (command.profile_id[ids] == 1)
      & (command.task_class[ids] == 3)
      & (command.cycle_phase[ids] == -1)
    )
    original = command.target_command[ids]
    positive = (
      (original[:, 0] > 0.04) & (original[:, 1] > 0.04) & (original[:, 2] < -0.04)
    )
    mirror = (
      (original[:, 0] < -0.04) & (original[:, 1] < -0.04) & (original[:, 2] > 0.04)
    )
    eligible &= (positive | mirror) & (original.abs().amax(-1) > 0.5)
    local = eligible.nonzero().flatten().cpu().tolist()
    if not local:
      return
    actual_ids = ids.cpu().tolist()
    original_rows = original.detach().cpu().tolist()
    new = []
    event = []
    for j in local:
      i = actual_ids[j]
      sign = 1 if original_rows[j][0] > 0 else -1
      name = "positive" if sign > 0 else "mirror"
      ordinal = self.counts[name]
      self.counts[name] += 1
      if ordinal % 2:
        continue
      variant = (ordinal // 2) % 4
      goal = [sign * v for v in POLICY["target"]]
      exact = variant in (0, 2)
      abrupt = variant in (0, 3)
      if not exact:
        goal = [v + self.rng.uniform(-0.05, 0.05) for v in goal]
      new.append((j, i, goal, abrupt))
      event.append(
        dict(
          env_id=i,
          step=int(step),
          kind="exact" if exact else "neighbor",
          sign=name,
          ordinal=ordinal,
          profile=1,
          phase=-1,
          abrupt=abrupt,
          original_target=original_rows[j],
          new_target=goal,
          source="actual command event only; teacher labels produced later on actual observations",
        )
      )
    if not new:
      return
    indices = torch.tensor([r[1] for r in new], device=ids.device, dtype=torch.long)
    goals = torch.tensor(
      [r[2] for r in new], device=original.device, dtype=original.dtype
    )
    abrupt = torch.tensor([r[3] for r in new], device=original.device, dtype=torch.bool)
    local_indices = torch.tensor(
      [r[0] for r in new], device=ids.device, dtype=torch.long
    )
    command.target_command[indices] = goals
    command.abrupt[indices] = abrupt
    command.vel_command_b[indices] = torch.where(
      abrupt[:, None], goals, before[local_indices]
    )
    self.events.extend(event)
    self.events = self.events[-128:]
    self.total_augmented += len(new)

  def state_dict(self):
    return copy.deepcopy(
      dict(
        version=1,
        policy=POLICY,
        rng=self.rng.getstate(),
        counts=self.counts,
        total_augmented=self.total_augmented,
        events=self.events,
      )
    )

  @classmethod
  def from_state(cls, saved):
    if saved["version"] != 1 or saved["policy"] != POLICY:
      raise ValueError("Unknown flat coverage policy")
    obj = cls()
    if set(saved["counts"]) != {"positive", "mirror"} or any(
      type(v) is not int or v < 0 for v in saved["counts"].values()
    ):
      raise ValueError("Invalid command event counters")
    if type(saved["total_augmented"]) is not int or saved["total_augmented"] < 0:
      raise ValueError("Invalid augmented event count")
    obj.counts = copy.deepcopy(saved["counts"])
    obj.total_augmented = saved["total_augmented"]
    if len(saved["events"]) > 128:
      raise ValueError("Bounded event ledger required")
    if len(saved["events"]) > saved["total_augmented"] or saved[
      "total_augmented"
    ] > sum(saved["counts"].values()):
      raise ValueError("Invalid event ledger totals")
    for row in saved["events"]:
      if (
        row["kind"] not in ("exact", "neighbor")
        or row["sign"] not in ("positive", "mirror")
        or type(row["abrupt"]) is not bool
      ):
        raise ValueError("Invalid target event kind")
      if (
        type(row["ordinal"]) is not int
        or row["ordinal"] < 0
        or type(row["step"]) is not int
        or row["step"] < 0
      ):
        raise ValueError("Invalid target event identity")
      if not 0 <= row["env_id"] < 1024 or row["phase"] != -1 or row["profile"] != 1:
        raise ValueError("Non-flat/mislabelled command event")
      if not all(math.isfinite(x) for x in row["new_target"]):
        raise ValueError("Nonfinite command target")
      sign = 1 if row["sign"] == "positive" else -1
      goal = [sign * v for v in POLICY["target"]]
      if (
        max(abs(a - b) for a, b in zip(row["new_target"], goal, strict=True))
        > 0.0500001
      ):
        raise ValueError("Target outside bounded command neighborhood")
    obj.events = copy.deepcopy(saved["events"])
    obj.rng.setstate(tuple_tree(saved["rng"]))
    return obj
