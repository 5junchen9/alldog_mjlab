"""Independent height/angle/direction curricula with conservative adjacent gates."""

import copy
import random
from typing import TypedDict


class GridPolicy(TypedDict):
  version: int
  minimum_trials: int
  window: int
  success_fraction: float
  both_directions_unlock: bool
  next_height_prerequisite_angles: list[int]
  birth_quota: list[str]
  private_CPU_seed: int
  maximum_exploration_cm: int


HEIGHTS = (4, 6, 8, 10, 12, 15, 20, 25, 30, 35)
ANGLES = (0, -15, 15, -30, 30, -45, 45)
POLICY: GridPolicy = dict(
  version=2,
  minimum_trials=32,
  window=64,
  success_fraction=0.90,
  both_directions_unlock=True,
  next_height_prerequisite_angles=[0, -15, 15],
  birth_quota=["F"] * 6 + ["R"] * 2 + ["M"] * 2,
  private_CPU_seed=181126001,
  maximum_exploration_cm=30,
)


LEGACY_POLICY = {**POLICY, "version": 1, "maximum_exploration_cm": 35}


def parse(key):
  try:
    height, angle, direction = (int(x) for x in key.split("/"))
  except (ValueError, AttributeError) as error:
    raise ValueError("Invalid grid cell") from error
  if (
    not 0 <= height < len(HEIGHTS)
    or angle not in ANGLES
    or direction not in (-1, 1)
    or key != f"{height}/{angle}/{direction}"
  ):
    raise ValueError("Unknown/noncanonical grid cell")
  return height, angle, direction


def key_for(height, angle, direction):
  if any(type(value) is not int for value in (height, angle, direction)):
    raise ValueError("Integer cell coordinates required")
  key = f"{height}/{angle}/{direction}"
  parse(key)
  return key


def passed(values):
  return len(values) >= 32 and sum(values) / len(values) >= 0.90


def checked_window(values):
  if (
    not isinstance(values, list)
    or len(values) > 64
    or any(type(value) is not bool for value in values)
  ):
    raise ValueError("Actual ordered boolean window of at most64 required")
  return list(values)


def tuple_tree(value):
  return (
    tuple(tuple_tree(x) for x in value) if isinstance(value, (list, tuple)) else value
  )


class GridCurriculum:
  def __init__(self):
    self.history = {}
    self.retained = set()
    self.mastery_evidence = {}
    self.last_birth = {}
    self.assigned_birth = {}
    self.record_counter = 0
    self.highest_sampled = 0
    self.birth_counter = 0
    self.quota_cursor = 0
    self.rng = random.Random(POLICY["private_CPU_seed"])
    self.members = {name: [] for name in ("F", "R", "M")}
    self.decks = {name: [] for name in ("F", "R", "M")}
    self.last_allocation = {}
    self.migration = None

  @classmethod
  def from_legacy(cls, history, height_level, angle_stage):
    if (
      type(height_level) is not int
      or not 0 <= height_level < len(HEIGHTS)
      or type(angle_stage) is not int
      or not 0 <= angle_stage < 4
    ):
      raise ValueError("Actual legacy height/angle stage required")
    value = cls()
    for key, window in history.items():
      if parse(key)[0] > height_level:
        raise ValueError("Legacy history exceeds actual course")
      value.history[key] = checked_window(window)
      value.retained.add(key)
      value._remember_mastery(key, "legacy_saved_window")
    enabled = ANGLES[: 1 + 2 * angle_stage]
    value.retained.update(
      key_for(height_level, angle, direction)
      for angle in enabled
      for direction in (-1, 1)
    )
    value.highest_sampled = height_level
    value.migration = dict(
      version=1,
      from_mode="coupled_height_and_angle",
      source_height_level=height_level,
      source_angle_stage=angle_stage,
      old_history_preserved=True,
      promotions_not_used_to_invent_mastery=True,
      legacy_enabled_only_grants_review_permission=True,
    )
    return value

  def mastered(self, height, angle, direction):
    return passed(self.history.get(key_for(height, angle, direction), []))

  def both(self, height, angle):
    return all(self.mastered(height, angle, direction) for direction in (-1, 1))

  def eligible(self, height, angle):
    key_for(height, angle, 1)
    if HEIGHTS[height] > POLICY["maximum_exploration_cm"]:
      return False
    if angle == 0:
      return height == 0 or all(self.both(height - 1, a) for a in (0, -15, 15))
    inner = 0 if abs(angle) == 15 else angle - (15 if angle > 0 else -15)
    return self.both(height, inner) and (height == 0 or self.both(height - 1, angle))

  def pools(self):
    frontier = set()
    for angle in ANGLES:
      candidates = [
        h
        for h in range(len(HEIGHTS))
        if self.eligible(h, angle)
        and any(not self.mastered(h, angle, d) for d in (-1, 1))
      ]
      if candidates:
        h = max(candidates)
        frontier.update(
          key_for(h, angle, d) for d in (-1, 1) if not self.mastered(h, angle, d)
        )
    active = {
      key
      for key in self.retained
      if HEIGHTS[parse(key)[0]] <= POLICY["maximum_exploration_cm"]
    }
    mastered = {key for key in active if passed(self.history.get(key, []))}
    recovery = active - mastered - frontier
    return {"F": sorted(frontier), "R": sorted(recovery), "M": sorted(mastered)}

  def sample(self, count):
    if type(count) is not int or count < 0:
      raise ValueError("Nonnegative explicit birth count required")
    pools = self.pools()
    if count and not any(pools.values()):
      raise ValueError("No legal course cell")
    for name, members in pools.items():
      if members != self.members[name]:
        self.members[name] = members
        self.decks[name] = []
    allocation = dict.fromkeys(("F", "R", "M"), 0)
    rows = []
    for _ in range(count):
      preferred = POLICY["birth_quota"][self.quota_cursor]
      self.quota_cursor = (self.quota_cursor + 1) % 10
      name = (
        preferred
        if pools[preferred]
        else next(key for key in ("F", "R", "M") if pools[key])
      )
      if not self.decks[name]:
        self.decks[name] = list(pools[name])
        self.rng.shuffle(self.decks[name])
      key = self.decks[name].pop()
      h, angle, direction = parse(key)
      if name == "F" and not self.eligible(h, angle):
        raise ValueError("New exploration lost prerequisites")
      rows.append((h, angle, direction))
      self.retained.add(key)
      self.highest_sampled = max(self.highest_sampled, h)
      self.birth_counter += 1
      allocation[name] += 1
    self.last_allocation = dict(
      count=count,
      actual=allocation,
      available={key: len(v) for key, v in pools.items()},
    )
    return rows

  def _remember_mastery(self, key, origin):
    values = self.history[key]
    if key not in self.mastery_evidence and passed(values):
      self.mastery_evidence[key] = dict(
        window=list(values), origin=origin, recorded_episode=self.record_counter
      )

  def record_many(self, rows):
    pending = []
    seen = set()
    pending_birth = dict(self.last_birth)
    for env_id, birth, height, angle, direction, success in rows:
      if type(env_id) is not int or not 0 <= env_id < 4096:
        raise ValueError("Actual formal environment id required")
      if type(birth) is not int or birth < 0 or type(success) is not bool:
        raise ValueError("Actual birth token and strict success boolean required")
      key = key_for(height, angle, direction)
      if key not in self.retained:
        raise ValueError("Episode route was not assigned or retained")
      previous = pending_birth.get(env_id, -1)
      if birth < previous:
        raise ValueError("Out-of-order episode birth")
      if birth == previous or (env_id, birth) in seen:
        continue
      seen.add((env_id, birth))
      pending_birth[env_id] = birth
      pending.append((env_id, birth, key, success))
    for env_id, birth, key, success in pending:
      self.last_birth[env_id] = birth
      values = self.history.setdefault(key, [])
      values.append(success)
      del values[:-64]
      self.retained.add(key)
      self.record_counter += 1
      self._remember_mastery(key, "recorded_strict_episode")
    return len(pending)

  def summary(self):
    pools = self.pools()
    cells = {}
    for key in sorted(self.retained):
      values = self.history.get(key, [])
      cells[key] = dict(
        trials=len(values),
        success=sum(values),
        currently_mastered=passed(values),
        once_mastered=key in self.mastery_evidence,
      )
    return dict(
      version=2,
      mode="independent_height_angle_direction",
      policy=POLICY,
      highest_sampled_level=self.highest_sampled,
      highest_sampled_cm=HEIGHTS[self.highest_sampled],
      cells=cells,
      available_pools={key: len(v) for key, v in pools.items()},
      last_allocation=self.last_allocation,
      birth_counter=self.birth_counter,
      training_mastery_is_not_certification=True,
    )

  def state_dict(self):
    return copy.deepcopy(
      dict(
        version=2,
        policy=POLICY,
        history=self.history,
        retained=sorted(self.retained),
        mastery_evidence=self.mastery_evidence,
        last_birth={str(k): v for k, v in self.last_birth.items()},
        assigned_birth={str(k): v for k, v in self.assigned_birth.items()},
        record_counter=self.record_counter,
        highest_sampled=self.highest_sampled,
        birth_counter=self.birth_counter,
        quota_cursor=self.quota_cursor,
        rng_state=self.rng.getstate(),
        members=self.members,
        decks=self.decks,
        last_allocation=self.last_allocation,
        migration=self.migration,
      )
    )

  @classmethod
  def from_state(cls, saved):
    legacy = type(saved.get("version")) is int and saved["version"] == 1
    if (
      type(saved.get("version")) is not int
      or saved["version"] not in (1, 2)
      or saved["policy"] != (LEGACY_POLICY if legacy else POLICY)
    ):
      raise ValueError("Grid policy/schema changed")
    value = cls()
    for key, window in saved["history"].items():
      parse(key)
      value.history[key] = checked_window(window)
    value.retained = set(saved["retained"])
    if (
      len(value.retained) != len(saved["retained"])
      or not value.history.keys() <= value.retained
    ):
      raise ValueError("Retained cell history missing/duplicated")
    for key in value.retained:
      parse(key)
    for name in (
      "record_counter",
      "birth_counter",
      "highest_sampled",
      "quota_cursor",
    ):
      if type(saved[name]) is not int or saved[name] < 0:
        raise ValueError("Invalid grid counter")
      setattr(value, name, saved[name])
    if value.highest_sampled >= len(HEIGHTS) or not 0 <= value.quota_cursor < 10:
      raise ValueError("Invalid sampled height/quota position")
    if any(parse(key)[0] > value.highest_sampled for key in value.retained):
      raise ValueError("Retained cell exceeds actual sampled source")
    value.mastery_evidence = copy.deepcopy(saved["mastery_evidence"])
    for key, proof in value.mastery_evidence.items():
      if key not in value.history or not passed(checked_window(proof["window"])):
        raise ValueError("No genuine window supports claimed past mastery")
    value.last_birth = {int(k): v for k, v in saved["last_birth"].items()}
    if any(
      not 0 <= k < 4096 or type(v) is not int or v < 0
      for k, v in value.last_birth.items()
    ):
      raise ValueError("Invalid per-environment birth token")
    value.assigned_birth = {int(k): v for k, v in saved["assigned_birth"].items()}
    if any(
      not 0 <= k < 4096 or type(v) is not int or v < 0
      for k, v in value.assigned_birth.items()
    ):
      raise ValueError("Invalid per-environment route assignment token")
    value.members = copy.deepcopy(saved["members"])
    value.decks = copy.deepcopy(saved["decks"])
    if set(value.members) != {"F", "R", "M"} or set(value.decks) != {"F", "R", "M"}:
      raise ValueError("Missing course pool/deck")
    for name in value.members:
      members, deck = value.members[name], value.decks[name]
      if (
        len(members) != len(set(members))
        or len(deck) != len(set(deck))
        or not set(deck) <= set(members)
      ):
        raise ValueError("Corrupt sampling deck")
      for key in members:
        parse(key)
    value.rng.setstate(tuple_tree(saved["rng_state"]))
    value.last_allocation = copy.deepcopy(saved["last_allocation"])
    value.migration = copy.deepcopy(saved["migration"])
    if legacy:
      value.migration = dict(
        kind="authorized_target35_to30",
        previous_migration=value.migration,
        history_and_mastery_preserved=True,
        RNG_and_counters_preserved=True,
        archived_above_target_not_sampled=True,
      )
      # Filter only scheduler entries outside the newly authorized target.
      # Existing <=30 decks retain order and the RNG is never advanced here.
      for name in value.members:
        value.members[name] = [
          key for key in value.members[name] if HEIGHTS[parse(key)[0]] <= 30
        ]
        value.decks[name] = [
          key for key in value.decks[name] if HEIGHTS[parse(key)[0]] <= 30
        ]
    elif any(
      HEIGHTS[parse(key)[0]] > 30
      for members in value.members.values()
      for key in members
    ):
      raise ValueError("Current scheduler exceeds authorized30cm target")
    return value
