"""Bounded dataset-aggregation BC with validation-driven student takeover.

Reward and privileged observations are never arguments to the student network.
No RSL PPO optimizer or default task is patched by this module.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DaggerConfig:
  loss: Literal["huber", "mse"] = "huber"
  huber_delta: float = 1.0
  dagger_enabled: bool = True
  learning_rate: float = 1e-4
  max_grad_norm: float = 1.0
  replay_capacity: int = 100_000
  batch_size: int = 1024
  updates_per_iteration: int = 4
  rollout_steps: int = 32
  max_iterations: int = 2000
  eval_interval: int = 25
  takeover_step: float = 0.1
  minimum_tracking_ratio: float = 0.9
  minimum_teacher_tracking: float = 0.1
  minimum_support_ratio: float = 0.9
  minimum_teacher_support: float = 0.5
  maximum_failure_rate: float = 0.05
  required_passes: int = 2
  seed: int = 101

  def __post_init__(self):
    if self.loss not in ("huber", "mse"):
      raise ValueError("loss must be huber or mse")
    for key in ("huber_delta", "learning_rate", "max_grad_norm"):
      v = getattr(self, key)
      if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{key} must be positive finite")
    for key in (
      "replay_capacity",
      "batch_size",
      "updates_per_iteration",
      "rollout_steps",
      "max_iterations",
      "eval_interval",
      "required_passes",
    ):
      if getattr(self, key) < 1:
        raise ValueError(f"{key} must be positive")
    for key in (
      "takeover_step",
      "minimum_tracking_ratio",
      "minimum_teacher_tracking",
      "maximum_failure_rate",
      "minimum_support_ratio",
      "minimum_teacher_support",
    ):
      if not 0 <= getattr(self, key) <= 1:
        raise ValueError(f"{key} must be within [0, 1]")
    if self.takeover_step == 0:
      raise ValueError("takeover_step must be positive")

  def validate_formal_run(self):
    if self.max_iterations < 2000:
      raise ValueError("formal training requires at least 2000 train iterations")


@dataclass(frozen=True)
class ValidationResult:
  tracking: float
  failure_rate: float
  episodes: int
  support_coverage: float | None = None

  def __post_init__(self):
    if self.episodes < 1:
      raise ValueError("validation must include completed episodes")
    if not 0 <= self.tracking <= 1 or not 0 <= self.failure_rate <= 1:
      raise ValueError("validation metrics must be finite within [0, 1]")
    if self.support_coverage is not None and not 0 <= self.support_coverage <= 1:
      raise ValueError("support coverage must be finite within [0, 1]")


class PerformanceGate:
  """Only separate, pure-student validation can increase takeover probability."""

  def __init__(self, cfg: DaggerConfig):
    self.cfg = cfg
    self.student_probability = 0.0
    self.pass_streak = 0

  def observe(self, student: ValidationResult, teacher: ValidationResult) -> float:
    if not self.cfg.dagger_enabled:
      self.student_probability = 0.0
      self.pass_streak = 0
      return 0.0
    if teacher.tracking < self.cfg.minimum_teacher_tracking:
      raise ValueError("teacher tracking is below the configured qualification floor")
    if teacher.failure_rate > self.cfg.maximum_failure_rate:
      raise ValueError("teacher failure rate is above the configured qualification cap")
    support_passed = True
    if (teacher.support_coverage is None) != (student.support_coverage is None):
      raise ValueError("teacher/student support evaluation contract differs")
    if teacher.support_coverage is not None:
      if teacher.support_coverage < self.cfg.minimum_teacher_support:
        raise ValueError("teacher support coverage is below qualification floor")
      assert student.support_coverage is not None
      support_passed = (
        student.support_coverage
        >= teacher.support_coverage * self.cfg.minimum_support_ratio
      )
    passed = (
      support_passed
      and student.tracking >= teacher.tracking * self.cfg.minimum_tracking_ratio
      and student.failure_rate <= self.cfg.maximum_failure_rate
    )
    if passed:
      self.pass_streak += 1
      if self.pass_streak >= self.cfg.required_passes:
        self.student_probability = min(
          1.0, self.student_probability + self.cfg.takeover_step
        )
        self.pass_streak = 0
    else:
      self.pass_streak = 0
      self.student_probability = max(
        0.0, self.student_probability - self.cfg.takeover_step
      )
    return self.student_probability


class DeploymentStudent(nn.Module):
  """Portable tensor-only actor. Input is actor history, never critic/velocity GT."""

  def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 128)):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dims = hidden_dims
    sizes = (input_dim, *hidden_dims, 16)
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
      layers.append(nn.Linear(a, b))
      if i < len(sizes) - 2:
        layers.append(nn.ELU())
    self.network = nn.Sequential(*layers)

  def forward(self, actor: torch.Tensor) -> torch.Tensor:
    if actor.ndim != 2 or actor.shape[-1] != self.input_dim:
      raise ValueError("student requires deployment actor tensor [B, input_dim]")
    return self.network(actor)


def student_input(observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
  """The single allowlist boundary. Never concatenate all observation groups."""
  actor = observations["actor"]
  if actor.ndim == 3:
    actor = actor.flip(1).flatten(1)  # exported contract: newest -> oldest.
  if actor.ndim != 2 or not torch.isfinite(actor).all():
    raise ValueError("actor observations must be finite [B, F] or [B, H, F]")
  return actor.detach().clone()


class Replay:
  """CPU FIFO aggregation across iterations, bounded for 8 GB GPU machines."""

  def __init__(self, capacity: int, input_dim: int):
    self.x = torch.empty(capacity, input_dim)
    self.y = torch.empty(capacity, 16)
    self.size = 0
    self.cursor = 0
    self.seen = 0

  def add(self, x: torch.Tensor, y: torch.Tensor):
    if x.shape[0] != y.shape[0] or y.shape[1:] != (16,):
      raise ValueError("replay batch shape mismatch")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
      raise ValueError("non-finite replay sample")
    n = x.shape[0]
    self.seen += n
    if n >= len(self.x):
      self.x.copy_(x[-len(self.x) :].detach().cpu())
      self.y.copy_(y[-len(self.y) :].detach().cpu())
      self.size, self.cursor = len(self.x), 0
      return
    ids = (torch.arange(n) + self.cursor) % len(self.x)
    self.x[ids] = x.detach().cpu()
    self.y[ids] = y.detach().cpu()
    self.cursor = (self.cursor + n) % len(self.x)
    self.size = min(len(self.x), self.size + n)

  def state_dict(self) -> dict[str, Any]:
    return {
      "x": self.x[: self.size].clone(),
      "y": self.y[: self.size].clone(),
      "size": self.size,
      "cursor": self.cursor,
      "seen": self.seen,
    }

  def load_state_dict(self, state: dict[str, Any]):
    size = state["size"]
    if not 0 <= size <= len(self.x):
      raise ValueError("replay capacity mismatch")
    self.x[:size].copy_(state["x"])
    self.y[:size].copy_(state["y"])
    self.size, self.cursor, self.seen = size, state["cursor"], state["seen"]


class DaggerTrainer:
  """Collect one state snapshot, label it, execute one policy, then aggregate."""

  def __init__(
    self,
    student: DeploymentStudent,
    teacher: nn.Module,
    cfg: DaggerConfig,
    teacher_group: Literal["actor", "critic"] = "actor",
  ):
    if teacher_group not in ("actor", "critic"):
      raise ValueError("invalid teacher group")
    self.student = student
    # Own a frozen copy; never share optimizer parameters or statistics.
    self.teacher = copy.deepcopy(teacher).eval().requires_grad_(False)
    student_ids = {id(p) for p in self.student.parameters()}
    if student_ids & {id(p) for p in self.teacher.parameters()}:
      raise ValueError("teacher/student must not share parameters")
    self.teacher_group = teacher_group
    self.cfg = cfg
    self.gate = PerformanceGate(cfg)
    self.device = next(student.parameters()).device
    self.teacher.to(self.device)
    self.optimizer = torch.optim.Adam(student.parameters(), lr=cfg.learning_rate)
    self.replay = Replay(cfg.replay_capacity, student.input_dim)
    self.rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
    self.iteration = 0
    self.optimizer_steps = 0
    self.transitions = 0

  @torch.no_grad()
  def collect(self, env, observations, steps: int | None = None):
    self.teacher.eval()
    self.student.eval()
    student_count = 0
    count = 0
    for _ in range(steps or self.cfg.rollout_steps):
      # Both input snapshots are taken before env.step. There is no second
      # observation_manager.compute call (which could advance history/noise).
      x = student_input(observations).to(self.device)
      teacher_x = observations[self.teacher_group].detach().clone().to(self.device)
      labels = self.teacher(teacher_x).detach()
      proposals = self.student(x)
      if labels.shape != proposals.shape or not torch.isfinite(proposals).all():
        raise ValueError("teacher/student action shape or finiteness mismatch")
      use_student = (
        torch.rand(x.shape[0], 1, generator=self.rng)
        < (self.gate.student_probability if self.cfg.dagger_enabled else 0.0)
      ).to(self.device)
      actions = torch.where(use_student, proposals, labels)
      self.replay.add(x, labels)  # only deployable inputs + expert action labels.
      observations, _, _, _ = env.step(actions)
      student_count += int(use_student.sum())
      count += x.shape[0]
    self.transitions += count
    return observations, {"samples": count, "student_samples": student_count}

  def update(self) -> dict[str, float]:
    if self.replay.size == 0:
      raise ValueError("collect data before updating")
    self.student.train()
    losses = []
    norm: float | torch.Tensor = 0.0
    for _ in range(self.cfg.updates_per_iteration):
      ids = torch.randint(self.replay.size, (self.cfg.batch_size,), generator=self.rng)
      x, y = self.replay.x[ids].to(self.device), self.replay.y[ids].to(self.device)
      prediction = self.student(x)
      loss = (
        F.huber_loss(prediction, y, delta=self.cfg.huber_delta)
        if self.cfg.loss == "huber"
        else F.mse_loss(prediction, y)
      )
      if not torch.isfinite(loss):
        raise FloatingPointError("non-finite distillation loss")
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      norm = nn.utils.clip_grad_norm_(
        self.student.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True
      )
      self.optimizer.step()
      self.optimizer_steps += 1
      losses.append(float(loss.detach()))
    self.iteration += 1
    return {"loss": sum(losses) / len(losses), "grad_norm": float(norm)}

  def state_dict(self) -> dict[str, Any]:
    return {
      "schema_version": 1,
      "config": asdict(self.cfg),
      "student": self.student.state_dict(),
      "optimizer": self.optimizer.state_dict(),
      "replay": self.replay.state_dict(),
      "rng": self.rng.get_state(),
      "iteration": self.iteration,
      "optimizer_steps": self.optimizer_steps,
      "transitions": self.transitions,
      "student_probability": self.gate.student_probability,
      "pass_streak": self.gate.pass_streak,
      "teacher_group": self.teacher_group,
    }

  def load_state_dict(self, state: dict[str, Any]):
    if (
      state["config"] != asdict(self.cfg)
      or state["teacher_group"] != self.teacher_group
    ):
      raise ValueError("resume configuration mismatch")
    if state.get("schema_version") != 1:
      raise ValueError("resume schema mismatch")
    self.student.load_state_dict(state["student"], strict=True)
    self.optimizer.load_state_dict(state["optimizer"])
    self.replay.load_state_dict(state["replay"])
    self.rng.set_state(state["rng"].cpu())
    for key in ("iteration", "optimizer_steps", "transitions"):
      setattr(self, key, state[key])
    self.gate.student_probability = state["student_probability"]
    self.gate.pass_streak = state["pass_streak"]
