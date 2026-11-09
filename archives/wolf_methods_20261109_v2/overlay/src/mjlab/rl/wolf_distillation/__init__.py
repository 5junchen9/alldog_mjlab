"""Opt-in Wolf BC/DAgger. Existing PPO/HIM tasks are unchanged."""

from .core import (
  DaggerConfig,
  DaggerTrainer,
  DeploymentStudent,
  PerformanceGate,
  ValidationResult,
)

__all__ = [
  "DaggerConfig",
  "DaggerTrainer",
  "DeploymentStudent",
  "PerformanceGate",
  "ValidationResult",
]
