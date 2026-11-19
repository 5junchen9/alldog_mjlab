"""Only withhold common positive rewards for prolonged unproductive stair motion."""

import copy
from typing import TypedDict

import torch


class GatePolicy(TypedDict):
  version: int
  dt: float
  window_steps: int
  startup_seconds: float
  external_push_grace_seconds: float
  demanded_progress_fraction: float
  progress_deadband_m: float
  maximum_credible_step_m: float
  moving_command_threshold: float
  positive_terms: list[str]
  no_extra_positive_reward: bool


CONFIG: GatePolicy = dict(
  version=1,
  dt=0.02,
  window_steps=100,
  startup_seconds=4.0,
  external_push_grace_seconds=1.0,
  demanded_progress_fraction=0.25,
  progress_deadband_m=0.005,
  maximum_credible_step_m=0.03,
  moving_command_threshold=0.05,
  positive_terms=["track_linear_velocity", "track_angular_velocity", "upright", "pose"],
  no_extra_positive_reward=True,
)


class ProgressCredit:
  scalar_fields = (
    "initialized",
    "age",
    "recovery_left",
    "raw_best",
    "previous",
    "credit",
    "demand",
    "gate",
    "window_credit",
    "window_demand",
    "qualified",
    "moving",
    "rejected_advance",
    "withheld_rate",
  )

  def __init__(self, n, device, num_levels=16):
    self.n = n
    self.num_levels = max(16, int(num_levels))
    self.last_step = -1
    self.cursor = 0
    self.migration = None
    for name in self.scalar_fields:
      boolean = name in ("initialized", "qualified", "moving", "rejected_advance")
      setattr(
        self,
        name,
        torch.zeros(n, device=device, dtype=torch.bool if boolean else torch.float32),
      )
    self.gate.fill_(1)
    self.credit_history = torch.zeros(100, n, device=device)
    self.demand_history = torch.zeros_like(self.credit_history)
    self.occupancy_seconds = torch.zeros(self.num_levels, 2, device=device)
    self.suppressed_seconds = torch.zeros_like(self.occupancy_seconds)

  def reset(self, ids):
    for name in self.scalar_fields:
      getattr(self, name)[ids] = 0
    self.gate[ids] = 1
    self.credit_history[:, ids] = 0
    self.demand_history[:, ids] = 0

  def on_external_push(self, ids):
    self.recovery_left[ids] = CONFIG["external_push_grace_seconds"]

  def prepare(
    self,
    step,
    dt,
    *,
    progress,
    command_xy,
    rough,
    safe,
    supported,
    corridor,
    levels,
    augmented,
  ):
    if step == self.last_step:
      return
    if step < self.last_step or abs(dt - CONFIG["dt"]) > 1e-7:
      raise ValueError("Progress credit needs the unchanged20ms control clock")
    self.last_step = step
    finite = torch.isfinite(progress) & torch.isfinite(command_xy).all(-1)
    position = torch.where(finite, progress, self.previous)
    fresh = ~self.initialized
    self.previous = torch.where(fresh, position, self.previous)
    self.raw_best = torch.where(fresh, position, self.raw_best)
    self.initialized |= rough
    advance = (position - self.raw_best).clamp(min=0)
    credible = safe & supported & corridor & finite
    credible &= (position - self.previous).abs() <= CONFIG["maximum_credible_step_m"]
    credible &= rough
    self.rejected_advance = rough & (advance > 0) & ~credible
    self.credit += torch.where(credible, advance, 0)
    # Invalid displacement is quarantined; it can never be credited next frame.
    self.raw_best = torch.maximum(self.raw_best, position)
    self.previous.copy_(position)
    speed = command_xy.norm(dim=-1)
    self.moving = rough & finite & (speed > CONFIG["moving_command_threshold"])
    self.demand += torch.where(self.moving, speed * dt, 0)
    self.window_credit = self.credit - self.credit_history[self.cursor]
    self.window_demand = self.demand - self.demand_history[self.cursor]
    self.credit_history[self.cursor].copy_(self.credit)
    self.demand_history[self.cursor].copy_(self.demand)
    self.cursor = (self.cursor + 1) % CONFIG["window_steps"]
    self.age += rough.float() * dt
    grace = (self.age <= CONFIG["startup_seconds"] + 1e-6) | (self.recovery_left > 0)
    fraction = (
      (self.window_credit - CONFIG["progress_deadband_m"]).clamp(min=0)
      / (CONFIG["demanded_progress_fraction"] * self.window_demand).clamp(min=1e-8)
    ).clamp(0, 1)
    self.gate = torch.where(self.moving & ~grace, fraction, 1.0)
    self.gate = torch.where(rough & ~finite, 0.0, self.gate)
    self.recovery_left = (self.recovery_left - dt).clamp(min=0)
    self.qualified = credible
    key = levels.clamp(0, self.num_levels - 1) * 2 + augmented.long()
    ids = rough.nonzero().flatten()
    self.occupancy_seconds.view(-1).index_add_(
      0, key[ids], torch.full_like(speed[ids], dt)
    )
    self.suppressed_seconds.view(-1).index_add_(
      0, key[ids], (self.gate[ids] < 0.999).float() * dt
    )

  def state_dict(self):
    return dict(
      config=copy.deepcopy(CONFIG),
      last_step=self.last_step,
      cursor=self.cursor,
      migration=self.migration,
      tensors={
        k: getattr(self, k).detach().cpu().clone()
        for k in (
          *self.scalar_fields,
          "credit_history",
          "demand_history",
          "occupancy_seconds",
          "suppressed_seconds",
        )
      },
    )

  def load_state_dict(self, saved):
    if saved is None:
      self.migration = "Fresh uninitialized progress account; no old learner/course/command/reward history changed"
      return
    if (
      saved["config"] != CONFIG
      or type(saved["cursor"]) is not int
      or not 0 <= saved["cursor"] < 100
    ):
      raise ValueError("Progress credit policy/schema mismatch")
    for name, value in saved["tensors"].items():
      target = getattr(self, name)
      if (
        value.shape != target.shape
        or value.dtype != target.dtype
        or not torch.isfinite(value).all()
      ):
        raise ValueError("Invalid progress credit tensor" + name)
      target.copy_(value.to(target.device))
    if set(saved["tensors"]) != set(self.state_dict()["tensors"]):
      raise ValueError("Incomplete progress account")
    if (
      (self.gate < 0).any()
      or (self.gate > 1).any()
      or (self.credit < 0).any()
      or (self.demand < 0).any()
    ):
      raise ValueError("Invalid progress gate/account range")
    self.last_step, self.cursor = saved["last_step"], saved["cursor"]
    self.migration = saved.get("migration")

  def summary(self):
    return dict(
      config=CONFIG,
      occupancy_seconds_by_height_and_route=self.occupancy_seconds.cpu().tolist(),
      suppressed_seconds_by_height_and_route=self.suppressed_seconds.cpu().tolist(),
      route_axis=["legacy_grid", "bounded_low_stair_path_v1"],
      current_gate_min=float(self.gate.min()),
      current_gate_mean=float(self.gate.mean()),
      no_learning_or_acceptance_claim=True,
    )


class ProgressDenseCost:
  def __init__(self, cfg, env):
    if cfg.weight != -1:
      raise ValueError("Only subtract withheld positive reward")
    self.env = env
    self.indices: list[int] | None = None

  def __call__(self, env):
    st = env.specialty446_state
    account = st.progress_credit
    if account.last_step != env.common_step_counter:
      raise RuntimeError("Task state must precede reward adjustment")
    manager = env.reward_manager
    if self.indices is None:
      names = manager.active_terms
      self.indices = [names.index(k) for k in CONFIG["positive_terms"]]
      if max(self.indices) >= names.index("specialty_progress_dense_cost"):
        raise ValueError("Original positive terms must be computed first")
    positive = manager._step_reward[:, self.indices].sum(-1)
    withheld = (1 - account.gate) * positive
    withheld = torch.where(st.command.flat_mask, 0.0, withheld)
    account.withheld_rate.copy_(withheld)
    return withheld

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.env.specialty446_state.progress_credit.reset(ids)
