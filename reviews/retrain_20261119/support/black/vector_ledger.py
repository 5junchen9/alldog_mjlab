"""Batched implementation of the reviewed passage state machine."""

import torch


class VectorLedger:
  fields = (
    "previous",
    "next_gate",
    "pending",
    "settled_steps",
    "closed",
    "pending_rate",
    "crossed_entry",
    "crossed_exit",
    "invalid",
    "progress_awarded",
    "terminal_awarded",
    "owed_ticks",
    "paid_ticks",
    "legacy_withheld_ticks",
  )

  def __init__(self, n, device, parameters):
    self.parameters = parameters
    self.previous = torch.zeros(n, device=device)
    self.next_gate = torch.zeros(n, device=device, dtype=torch.long)
    self.pending = torch.zeros(n, device=device)
    self.settled_steps = torch.zeros(n, device=device, dtype=torch.long)
    self.closed = torch.zeros(n, device=device, dtype=torch.bool)
    self.pending_rate = torch.zeros(n, device=device)
    self.crossed_entry = torch.zeros(n, device=device, dtype=torch.bool)
    self.crossed_exit = torch.zeros(n, device=device, dtype=torch.bool)
    self.invalid = torch.zeros(n, device=device, dtype=torch.bool)
    self.progress_awarded = torch.zeros(n, device=device, dtype=torch.bool)
    self.terminal_awarded = torch.zeros_like(self.progress_awarded)
    self.owed_ticks = torch.zeros(n, device=device, dtype=torch.long)
    self.paid_ticks = torch.zeros_like(self.owed_ticks)
    self.legacy_withheld_ticks = torch.zeros_like(self.owed_ticks)
    self.migration = None
    self.last_step = -1
    self.paid_step = -1

  def prepare(
    self,
    step,
    dt,
    progress,
    gates,
    active,
    supported,
    settled,
    failed,
    valid,
    *,
    exit_safe=None,
  ):
    if self.last_step == step:
      return
    if step < self.last_step or dt <= 0:
      raise ValueError("Invalid ledger clock")
    self.last_step = step
    self.pending_rate.zero_()
    live = active & ~self.closed
    invalid = (
      failed
      | ~valid
      | ~torch.isfinite(progress)
      | ((progress - self.previous).abs() > self.parameters["maximum_step_progress"])
    )
    invalid |= (self.next_gate == 0) & (progress >= gates[:, 1])
    invalid &= live
    self.invalid |= invalid
    self.closed |= invalid
    self.pending[invalid] = 0
    self.owed_ticks[invalid] = 0
    self.settled_steps[invalid] = 0
    live &= ~invalid
    # Crossing is latched independently of the instantaneous gait contacts.
    # Confirmation may arrive on a later frame, still beyond that ordered gate.
    self.crossed_entry |= (
      live
      & (self.next_gate == 0)
      & (self.previous < gates[:, 0])
      & (progress >= gates[:, 0])
    )
    self.crossed_exit |= (
      live
      & (self.next_gate == 1)
      & (self.previous < gates[:, 1])
      & (progress >= gates[:, 1])
    )
    first = (
      live
      & (self.next_gate == 0)
      & self.crossed_entry
      & (progress >= gates[:, 0])
      & supported
    )
    second = (
      live
      & (self.next_gate == 1)
      & self.crossed_exit
      & (progress >= gates[:, 1])
      & supported
    )
    credits = self.parameters["stage_credits"]
    self.pending += first * credits[0] + second * credits[1]
    self.next_gate += first.long() + second.long()
    self.previous = torch.where(live, progress, self.previous)
    stable = (
      live & (self.next_gate == 2) & (progress >= gates[:, 1]) & supported & settled
    )
    self.settled_steps = torch.where(
      stable, self.settled_steps + 1, torch.where(live, 0, self.settled_steps)
    )
    finished = live & (self.settled_steps >= self.parameters["settle_steps"])
    # Only payment timing changes. The original finished/invalid/closed masks
    # and all curriculum predicates above remain unchanged.
    if exit_safe is None:
      exit_safe = torch.zeros_like(active)  # absent physical proof never pays
    passed_exit = (
      live
      & (self.next_gate == 2)
      & supported
      & exit_safe
      & (progress >= gates[:, 1])
      & ~self.progress_awarded
    )
    self.owed_ticks += passed_exit.long()  # 1 quarter-unit, once per episode
    self.progress_awarded |= passed_exit
    terminal = finished & exit_safe & ~self.terminal_awarded
    # Migration retires the small credit for every legacy in-flight episode;
    # fresh physical reset clears this conservative withheld marker.
    self.owed_ticks += terminal.long() * 11  # strict completion2.75
    self.terminal_awarded |= terminal
    self.closed |= finished
    self.pending_rate = self.owed_ticks.to(progress.dtype) / (4 * dt)

  def consume(self, step):
    if step != self.last_step:
      raise RuntimeError("Termination must prepare ledger before reward")
    if self.paid_step == step:
      return torch.zeros_like(self.pending_rate)
    self.paid_step = step
    reward = self.pending_rate.clone()
    self.paid_ticks += self.owed_ticks
    self.owed_ticks.zero_()
    return reward

  def reset(self, ids):
    for name in self.fields:
      getattr(self, name)[ids] = 0

  def state_dict(self):
    return dict(
      version=3,
      payment_schema="safe_full_exit_025_strict_finish_275_v1",
      migration=self.migration,
      parameters=self.parameters,
      last_step=self.last_step,
      paid_step=self.paid_step,
      tensors={
        name: getattr(self, name).detach().cpu().clone() for name in self.fields
      },
    )

  def load_state_dict(self, saved):
    if saved["version"] not in (2, 3) or saved["parameters"] != self.parameters:
      raise ValueError("Ledger schema/config changed")
    old_fields = self.fields[:9]
    names = old_fields if saved["version"] == 2 else self.fields
    if saved["version"] == 3 and saved.get("payment_schema") != (
      "safe_full_exit_025_strict_finish_275_v1"
    ):
      raise ValueError("Wrong reward payment schema")
    for name in names:
      actual = getattr(self, name)
      value = saved["tensors"][name]
      if (
        value.shape != actual.shape
        or value.dtype != actual.dtype
        or not torch.isfinite(value).all()
      ):
        raise ValueError("Invalid ledger tensor " + name)
      actual.copy_(value.to(actual.device))
    self.last_step = saved["last_step"]
    self.paid_step = saved["paid_step"]
    if saved["version"] == 2:
      # There is no per-event payment history in v2. Never turn its restored
      # position or pending final rate into a new reward. Fresh episodes reset
      # these markers after the complete learner/course restore is verified.
      completed = (
        self.closed
        & ~self.invalid
        & (self.settled_steps >= self.parameters["settle_steps"])
      )
      self.progress_awarded.fill_(True)
      self.terminal_awarded.copy_(completed)
      self.owed_ticks.zero_()
      self.paid_ticks.copy_(completed.long() * 12)
      self.legacy_withheld_ticks.copy_((~completed).long())
      self.pending_rate.zero_()
      self.migration = dict(
        from_version=2,
        retroactive_small_reward=False,
        completed_events_retired=True,
        legacy_pending_rate_discarded=True,
        open_episode_small_credit_withheld=True,
        fresh_physical_reset_clears_withheld=True,
      )
    else:
      self.migration = saved.get("migration")
    if (
      (self.owed_ticks < 0).any()
      or (self.paid_ticks < 0).any()
      or (self.legacy_withheld_ticks < 0).any()
      or (self.legacy_withheld_ticks > 1).any()
      or ((self.owed_ticks + self.paid_ticks + self.legacy_withheld_ticks) > 12).any()
    ):
      raise ValueError("Saved event budget exceeds3 or is negative")
