"""Robot-independent HIM spec interfaces and computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HIMSpec:
  single_frame_dim: int
  history_length: int
  command_dim: int
  velocity_dim: int
  action_dim: int
  latent_dim: int

  @property
  def history_dim(self) -> int:
    return self.history_length * self.single_frame_dim

  @property
  def encoder_input_dim(self) -> int:
    return self.history_length * (self.single_frame_dim - self.command_dim)

  @property
  def target_encoder_input_dim(self) -> int:
    return self.single_frame_dim - self.command_dim + self.velocity_dim

  @property
  def actor_input_dim(self) -> int:
    return self.single_frame_dim + self.velocity_dim + self.latent_dim


def canonical_history(history: torch.Tensor) -> torch.Tensor:
  return history.flip(1).flatten(1)


def encoder_history(
  history: torch.Tensor,
  history_length: int,
  single_frame_dim: int,
  command_dim: int,
) -> torch.Tensor:
  """Remove each frame's command only at the source encoder boundary.

  Input retains the complete canonical newest-to-oldest history used by play,
  storage and deployment. Actor observations and target inputs are unchanged.
  """
  frames = history.reshape(-1, history_length, single_frame_dim)
  return frames[..., command_dim:].flatten(1)


def estimator_target_input(
  frame: torch.Tensor, velocity: torch.Tensor, spec: HIMSpec
) -> torch.Tensor:
  return torch.cat((frame[..., spec.command_dim :], velocity), dim=-1)
