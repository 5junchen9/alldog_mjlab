"""Robot-independent HIM estimator interfaces and computation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import MLP

from .spec import HIMSpec, encoder_history


class HIMEstimator(nn.Module):
  def __init__(
    self,
    spec: HIMSpec,
    *,
    encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
    target_encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
    activation: str = "elu",
    num_prototypes: int = 32,
    temperature: float = 3.0,
  ) -> None:
    super().__init__()
    if len(encoder_hidden_dims) < 1 or len(target_encoder_hidden_dims) < 1:
      raise ValueError(
        "encoder_hidden_dims and target_encoder_hidden_dims must be non-empty."
      )
    self.spec = spec
    self.temperature = temperature
    self.num_prototypes = num_prototypes
    self.encoder = MLP(
      spec.encoder_input_dim,
      spec.velocity_dim + spec.latent_dim,
      hidden_dims=list(encoder_hidden_dims),
      activation=activation,
    )
    self.target = MLP(
      spec.target_encoder_input_dim,
      spec.latent_dim,
      hidden_dims=list(target_encoder_hidden_dims),
      activation=activation,
    )
    self.proto = nn.Embedding(num_prototypes, spec.latent_dim)

  def encode(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoder_input = encoder_history(
      history,
      self.spec.history_length,
      self.spec.single_frame_dim,
      self.spec.command_dim,
    )
    parts = self.encoder(encoder_input)
    velocity = parts[..., : self.spec.velocity_dim]
    latent = F.normalize(parts[..., self.spec.velocity_dim :], dim=-1, p=2)
    return (velocity, latent)

  def encode_target(self, next_input: torch.Tensor) -> torch.Tensor:
    return F.normalize(self.target(next_input), dim=-1, p=2)

  @torch.no_grad()
  def predict(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.encode(history)

  def losses(
    self,
    history: torch.Tensor,
    next_input: torch.Tensor,
    next_velocity: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    velocity, source_latent = self.encode(history)
    target_latent = self.encode_target(next_input)
    with torch.no_grad():
      normalized_proto = F.normalize(self.proto.weight.data.clone(), dim=-1, p=2)
      self.proto.weight.copy_(normalized_proto)
    score_s = source_latent @ self.proto.weight.T
    score_t = target_latent @ self.proto.weight.T
    with torch.no_grad():
      q_s = sinkhorn(score_s)
      q_t = sinkhorn(score_t)
    log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
    log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)
    swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
    estimation_loss = F.huber_loss(velocity, next_velocity, delta=1.0)
    return (estimation_loss, swap_loss)


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
  Q = torch.exp(out / eps).T
  K, B = (Q.shape[0], Q.shape[1])
  Q /= Q.sum()
  for _ in range(iters):
    Q /= torch.sum(Q, dim=1, keepdim=True)
    Q /= K
    Q /= torch.sum(Q, dim=0, keepdim=True)
    Q /= B
  return (Q * B).T
