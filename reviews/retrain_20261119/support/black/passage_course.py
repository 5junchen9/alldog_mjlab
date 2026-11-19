"""Course-only staircase upgrades. Neither rewards nor strict evaluator consume this state."""

from __future__ import annotations

import collections
import copy
import random

SCHEMA = "black_stairs_course_passage_v1"
HEIGHTS = (4, 6, 8, 10, 12, 15, 20, 25, 30, 35)
FROZEN = "408ef2af0adb8f5c00ddf9d5094ec2885b48875bb867f5256283675c0b35c1fa"
POLICY = dict(
  minimum_trials=32,
  window=64,
  pass_rate=0.90,
  maximum_cm=30,
  headon=0.90,
  lower_review=0.20,
  auxiliary_angles=(-15, 15),
  seed=181127441,
)
SUCCESS = {"pass_strict_stop", "pass_timeout_no_stop"}
RATED = SUCCESS | {
  "failure",
  "timeout_no_pass",
  "strict_stop_without_pass",
  "evidence_invalid",
}


def typed_equal(a, b):
  if type(a) is not type(b):
    return False
  if isinstance(a, dict):
    return a.keys() == b.keys() and all(typed_equal(a[k], b[k]) for k in a)
  if isinstance(a, (list, tuple)):
    return len(a) == len(b) and all(
      typed_equal(x, y) for x, y in zip(a, b, strict=True)
    )
  return a == b


def schedule_classify(pass_completed, failed, timeout, strict, evidence_invalid=False):
  if failed:
    return "failure"
  if evidence_invalid:
    return "evidence_invalid"
  if strict:
    return "pass_strict_stop" if pass_completed else "strict_stop_without_pass"
  if timeout:
    return "pass_timeout_no_stop" if pass_completed else "timeout_no_pass"
  return "censored"


class PassageCourse:
  def __init__(self, policy=None):
    self.policy = copy.deepcopy(POLICY if policy is None else policy)
    p = self.policy
    if set(p) != set(POLICY) or any(type(p[k]) is not type(POLICY[k]) for k in POLICY):
      raise ValueError("noncanonical schema")
    if (
      p["minimum_trials"] < 1
      or not p["minimum_trials"] <= p["window"]
      or not 0 < p["pass_rate"] <= 1
    ):
      raise ValueError("bad evidence policy")
    if (
      p["maximum_cm"] not in HEIGHTS
      or not 0 < p["headon"] <= 1
      or not 0 <= p["lower_review"] < 1
    ):
      raise ValueError("bad route policy")
    if not p["auxiliary_angles"] or not set(p["auxiliary_angles"]) <= {
      -45,
      -30,
      -15,
      15,
      30,
      45,
    }:
      raise ValueError("bad auxiliary set")
    self.active = {}
    self.last_closed = {}
    self.current = 0
    self.births = 0
    self.rng = random.Random(p["seed"])
    self.history = collections.defaultdict(list)
    self.outcomes = collections.Counter()
    self.promotion = []
    self.votes = 0
    self.assigned = collections.Counter()
    self.cold_censored = 0
    self.proof_tail = []

  def assign(self, lane, stamp):
    last = self.active.get(lane)
    if last is not None:
      if stamp == last[0]:
        return last[1:]
      raise RuntimeError("old episode must be classified before new assignment")
    if stamp <= self.last_closed.get(lane, -1):
      raise ValueError("birth clock backwards")
    p = self.policy
    h = self.current
    if h > 0 and self.rng.random() < p["lower_review"]:
      h = self.rng.randrange(h)
    a = 0 if self.rng.random() < p["headon"] else self.rng.choice(p["auxiliary_angles"])
    # Separate direction clocks so even sparse auxiliary or low review has no fixed bias.
    parity = self.assigned[h, a]
    direction = (-1, 1)[parity % 2]
    self.assigned[h, a] += 1
    row = (h, a, direction)
    self.active[lane] = (stamp, *row)
    self.births += 1
    return row

  def vote(self, lane, stamp, level, angle, direction, outcome, proof=None):
    if outcome not in RATED | {"censored"}:
      raise ValueError("unknown episode outcome")
    key = (stamp, level, angle, direction)
    if stamp <= self.last_closed.get(lane, -1):
      return False
    if self.active.get(lane) != key:
      return False  # archived or external route cannot vote
    if outcome in SUCCESS and (
      not isinstance(proof, dict)
      or proof.get("ordered_treads") != 12
      or proof.get("landing_contact") is not True
      or not stamp
      < proof.get("first_verified_pass_tick", -1)
      <= proof.get("closed_tick", -1)
    ):
      raise ValueError("A claimed pass needs a real first witness")
    self.last_closed[lane] = stamp
    del self.active[lane]
    self.outcomes[outcome] += 1
    self.proof_tail.append(
      dict(
        lane=lane,
        birth=stamp,
        level=level,
        angle=angle,
        direction=direction,
        outcome=outcome,
        proof=copy.deepcopy(proof),
      )
    )
    del self.proof_tail[:-256]
    if angle == 0 and outcome in RATED:
      slots = self.history[level, direction]
      slots.append(outcome)
      del slots[: -self.policy["window"]]
      self.votes += 1
    return True

  def advance_once(self, stamp):
    p = self.policy
    h = self.current
    if h >= HEIGHTS.index(p["maximum_cm"]):
      return False
    if self.promotion and self.promotion[-1]["step"] >= stamp:
      return False
    if self.proof_tail and any(
      (r["proof"] or {}).get("closed_tick", -1) > stamp for r in self.proof_tail
    ):
      raise ValueError("course clock before observed episode closure")
    score = {d: list(self.history[h, d]) for d in (-1, 1)}
    if any(
      len(v) < p["minimum_trials"]
      or sum(x in SUCCESS for x in v) / len(v) < p["pass_rate"]
      for v in score.values()
    ):
      return False
    self.current += 1
    self.promotion.append(
      dict(
        step=stamp,
        before=h,
        after=self.current,
        vote_windows={str(d): v for d, v in score.items()},
      )
    )
    return True

  def state_dict(self):
    return copy.deepcopy(
      dict(
        schema=SCHEMA,
        frozen=FROZEN,
        policy=self.policy,
        current=self.current,
        history=[[h, d, r] for (h, d), r in sorted(self.history.items())],
        outcomes=dict(self.outcomes),
        promotion=self.promotion,
        votes=self.votes,
        births=self.births,
        last_closed=self.last_closed,
        active=self.active,
        assigned=[[h, a, n] for (h, a), n in sorted(self.assigned.items())],
        rng=self.rng.getstate(),
        cold_censored=self.cold_censored,
        proof_tail=self.proof_tail,
      )
    )

  @classmethod
  def restore(cls, p, policy, *, cold_reset, explicit_candidate):
    if (
      not explicit_candidate
      or p.get("schema") != SCHEMA
      or p.get("frozen") != FROZEN
      or not typed_equal(p.get("policy"), policy)
    ):
      raise ValueError("Not a saved passage-course generation; legacy hot load refused")
    z = cls(policy)
    assert type(p["current"]) is int and 0 <= p["current"] <= HEIGHTS.index(
      policy["maximum_cm"]
    )
    z.current = p["current"]
    z.births = p["births"]
    z.votes = p["votes"]
    z.cold_censored = p["cold_censored"]
    for h, d, r in p["history"]:
      assert (
        type(h) is int
        and 0 <= h <= z.current
        and type(d) is int
        and d in (-1, 1)
        and 0 <= len(r) <= policy["window"]
      )
      assert all(x in RATED for x in r) and (h, d) not in z.history
      z.history[h, d] = list(r)
    z.promotion = copy.deepcopy(p["promotion"])
    assert len(z.promotion) == z.current
    assert all(
      x["before"] == i
      and x["after"] == i + 1
      and (i == 0 or x["step"] > z.promotion[i - 1]["step"])
      for i, x in enumerate(z.promotion)
    )
    for item in z.promotion:
      assert set(item["vote_windows"]) == {"-1", "1"}
      for window in item["vote_windows"].values():
        assert policy["minimum_trials"] <= len(window) <= policy["window"]
        assert all(x in RATED for x in window)
        assert sum(x in SUCCESS for x in window) / len(window) >= policy["pass_rate"]
    z.outcomes = collections.Counter(p["outcomes"])
    assert all(
      k in RATED | {"censored"} and type(v) is int and v >= 0
      for k, v in z.outcomes.items()
    )
    z.proof_tail = copy.deepcopy(p["proof_tail"])
    assert len(z.proof_tail) <= 256 and all(
      r["outcome"] in RATED | {"censored"} for r in z.proof_tail
    )
    for r in z.proof_tail:
      if r["outcome"] in SUCCESS:
        assert (
          r["proof"]["ordered_treads"] == 12 and r["proof"]["landing_contact"] is True
        )
        assert (
          r["birth"]
          < r["proof"]["first_verified_pass_tick"]
          <= r["proof"]["closed_tick"]
        )
    z.last_closed = copy.deepcopy(p["last_closed"])
    z.active = copy.deepcopy(p["active"])
    assert all(
      type(k) is int and k >= 0 and type(v) is int and v >= 0
      for k, v in z.last_closed.items()
    )
    for lane, b in z.active.items():
      assert (
        type(lane) is int
        and lane >= 0
        and type(b) is tuple
        and len(b) == 4
        and all(type(v) is int for v in b)
      )
      assert (
        b[0] > z.last_closed.get(lane, -1)
        and 0 <= b[1] <= z.current
        and b[2] in (0, *policy["auxiliary_angles"])
        and b[3] in (-1, 1)
      )
    if cold_reset:
      for i, data in z.active.items():
        z.last_closed[i] = data[0]
        z.cold_censored += 1
      z.active.clear()
    assert all(type(v) is int and v >= 0 for v in (z.births, z.votes, z.cold_censored))
    assert z.births == sum(z.outcomes.values()) + len(z.active) + z.cold_censored
    assert z.votes <= sum(z.outcomes[k] for k in RATED)
    for h, a, n in p["assigned"]:
      assert (
        type(h) is int
        and 0 <= h <= z.current
        and type(a) is int
        and a in (0, *policy["auxiliary_angles"])
        and type(n) is int
        and n >= 0
      )
    z.assigned = collections.Counter({(h, a): n for h, a, n in p["assigned"]})
    assert (
      len(z.assigned) == len(p["assigned"]) and sum(z.assigned.values()) == z.births
    )
    z.rng.setstate(p["rng"])
    return z
