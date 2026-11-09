"""Small GLAD/WM-LOCO-inspired adapters, not reproductions of either paper.

The deployable student is BLIND: it consumes proprioceptive history, not images.
A two-scale global/local attention teacher supplies training-only targets. The
predictive model learns action-conditioned ONE-STEP proprioception, not a latent
world model, imagined rollout, camera reconstruction, or model-based planner.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.modules import MLP, HiddenState
from torch import nn
from torch.nn import functional as F

from .policy import HIMPolicy
from .spec import canonical_history


class HistoryFeatures(nn.Module):
  """Independent switches remove each feature branch and its parameters."""

  def __init__(
    self, history_dim: int, latent_dim: int, *, distillation: bool, prediction: bool
  ) -> None:
    super().__init__()
    self.distillation = distillation
    self.prediction = prediction
    self.terrain = (
      MLP(history_dim, latent_dim, [128, 64], "elu") if distillation else nn.Identity()
    )
    self.predictive = (
      MLP(history_dim, latent_dim, [128, 64], "elu") if prediction else nn.Identity()
    )
    self.output_dim = latent_dim * (int(distillation) + int(prediction))

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    parts = [history[:, :0]]
    if self.distillation:
      parts.append(torch.tanh(self.terrain(history)))
    if self.prediction:
      parts.append(torch.tanh(self.predictive(history)))
    return torch.cat(parts, dim=-1)


class ScanAttentionEncoder(nn.Module):
  """Global pooling or state-conditioned sparse local attention over ray tokens.

  Input packs x/y/clearance/valid in four contiguous channels. Invalid rays have
  exactly zero attention, including the all-missing case. Dot-product scores
  depend on proprioception for local top-K selection (not an additive bias).
  """

  def __init__(
    self, input_dim: int, history_dim: int, latent_dim: int, *, sparse: bool
  ):
    super().__init__()
    if input_dim <= 0 or input_dim % 4:
      raise ValueError("Teacher scan must contain four channels per ray")
    self.num_rays = input_dim // 4
    self.sparse = sparse
    self.top_k = min(8, self.num_rays)
    self.tokens = MLP(3, 32, [32], "elu")
    self.query = nn.Linear(history_dim, 32) if sparse else None
    self.global_query = nn.Parameter(torch.randn(32) * 0.1)
    self.output = nn.Linear(32, latent_dim)

  def forward(self, scan: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    points = scan.reshape(-1, 4, self.num_rays).transpose(1, 2)
    valid = points[..., 3] > 0.5
    tokens = self.tokens(points[..., :3])
    query = (
      self.query(history)
      if self.query is not None
      else self.global_query.expand(scan.shape[0], -1)
    )
    scores = (tokens * query.unsqueeze(1)).sum(-1) / (32.0**0.5)
    scores = scores.masked_fill(~valid, -1e4)
    if self.sparse:
      selected = scores.topk(self.top_k, dim=-1).indices
      mask = torch.zeros_like(valid).scatter(1, selected, True)
      valid = valid & mask
      scores = scores.masked_fill(~valid, -1e4)
    weights = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    pooled = (tokens * weights.unsqueeze(-1)).sum(1)
    return self.output(pooled)


class PerceptionPredictionAux(nn.Module):
  """Privileged terrain enters losses only; never the student forward method."""

  def __init__(
    self,
    history_dim: int,
    action_dim: int,
    target_dim: int,
    local_dim: int,
    global_dim: int,
    latent_dim: int = 16,
    *,
    distillation: bool = True,
    prediction: bool = True,
  ) -> None:
    super().__init__()
    self.distillation = distillation
    self.prediction = prediction
    self.latent_dim = latent_dim
    self.target_dim = target_dim
    self.features = HistoryFeatures(
      history_dim, latent_dim, distillation=distillation, prediction=prediction
    )
    if distillation:
      if local_dim <= 0 or global_dim <= 0:
        raise ValueError("Terrain distillation requires both privileged scan groups")
      self.teacher_local = ScanAttentionEncoder(
        local_dim, history_dim, latent_dim, sparse=True
      )
      self.teacher_global = ScanAttentionEncoder(
        global_dim, history_dim, latent_dim, sparse=False
      )
      self.teacher_fuse = MLP(2 * latent_dim, latent_dim, [64], "elu")
      self.terrain_decoder = MLP(latent_dim, local_dim + global_dim, [64], "elu")
    if prediction:
      self.predictor = MLP(latent_dim + action_dim, target_dim, [64], "elu")

  def losses(
    self,
    history: torch.Tensor,
    actions: torch.Tensor,
    next_frame: torch.Tensor,
    local_scan: torch.Tensor | None = None,
    global_scan: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    features = self.features(history)
    losses: dict[str, torch.Tensor] = {}
    if self.distillation:
      if local_scan is None or global_scan is None:
        raise ValueError("Missing training-only terrain teacher inputs")
      teacher = torch.tanh(
        self.teacher_fuse(
          torch.cat(
            (
              self.teacher_local(local_scan, history),
              self.teacher_global(global_scan, history),
            ),
            dim=-1,
          )
        )
      )
      # Reconstruction trains the teacher independently of the student, rather
      # than allowing two freely moving encoders to minimize a constant latent.
      losses["terrain_reconstruction"] = F.smooth_l1_loss(
        self.terrain_decoder(teacher), torch.cat((local_scan, global_scan), dim=-1)
      )
      losses["terrain_distillation"] = F.mse_loss(
        features[:, : self.latent_dim], teacher.detach()
      )
    if self.prediction:
      latent = features[:, -self.latent_dim :]
      prediction = self.predictor(torch.cat((latent, actions), dim=-1))
      losses["next_proprioception"] = F.smooth_l1_loss(
        prediction, next_frame[:, : self.target_dim].detach()
      )
    return losses


class PerceptionHIMPolicy(HIMPolicy):
  """HIM plus an actor residual driven exclusively by deployable history."""

  def __init__(
    self,
    obs,
    *args,
    use_him_estimator: bool = True,
    distillation_enabled: bool = True,
    prediction_enabled: bool = True,
    methods_latent_dim: int = 16,
    teacher_local_group: str = "terrain_teacher_local",
    teacher_global_group: str = "terrain_teacher_global",
    **kwargs,
  ) -> None:
    super().__init__(obs, *args, **kwargs)
    self.use_him_estimator = use_him_estimator
    if not use_him_estimator:
      self.estimator.requires_grad_(False)
    self.teacher_local_group = teacher_local_group
    self.teacher_global_group = teacher_global_group
    # Predict physical proprioception, excluding commands and the trivial
    # previous-action channels. The actor's original frame contract is intact.
    target_dim = (
      self.spec.single_frame_dim - self.spec.command_dim - self.spec.action_dim
    )
    if target_dim <= 0:
      raise ValueError("Prediction requires non-command/non-action observations")
    self.auxiliary = PerceptionPredictionAux(
      self.spec.history_dim,
      self.spec.action_dim,
      target_dim,
      obs[teacher_local_group].shape[-1] if distillation_enabled else 0,
      obs[teacher_global_group].shape[-1] if distillation_enabled else 0,
      methods_latent_dim,
      distillation=distillation_enabled,
      prediction=prediction_enabled,
    )
    self.methods_enabled = distillation_enabled or prediction_enabled
    output_layer = self.mlp[-1]
    assert isinstance(output_layer, nn.Linear)
    self.residual = (
      nn.Linear(self.auxiliary.features.output_dim, output_layer.out_features)
      if self.methods_enabled
      else None
    )
    if self.residual is not None:
      nn.init.zeros_(self.residual.weight)
      nn.init.zeros_(self.residual.bias)

  def get_latent(
    self, obs, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
  ) -> torch.Tensor:
    if self.use_him_estimator:
      return super().get_latent(obs, masks, hidden_state)
    history = obs[self.history_group]
    if history.ndim != 3:
      raise ValueError("HIM-compatible input must retain [B, H, F] history")
    current = history[..., -1, :]
    # Preserve the actor MLP shape/budget for a controlled mechanism ablation.
    # No estimator inference, true velocity, or old frame enters this base path.
    zeros = current.new_zeros(
      (current.shape[0], self.spec.velocity_dim + self.spec.latent_dim)
    )
    return torch.cat((current, zeros), dim=-1)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
    output = self.mlp(self.get_latent(obs, masks, hidden_state))
    if self.residual is not None:
      # Auxiliary encoders are trained separately after PPO, never by the
      # privileged critic or the actor loss. Actor residual weights use PPO.
      with torch.no_grad():
        features = self.auxiliary.features(canonical_history(obs[self.history_group]))
      output = output + self.residual(features)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(output)
    return output

  def policy_parameters(self):
    yield from super().policy_parameters()
    if self.residual is not None:
      yield from self.residual.parameters()

  def as_jit(self) -> nn.Module:
    return _PerceptionActorJit(self)


class _PerceptionActorJit(nn.Module):
  def __init__(self, policy: PerceptionHIMPolicy) -> None:
    super().__init__()
    self.history_dim = policy.spec.history_dim
    self.history_length = policy.spec.history_length
    self.frame_dim = policy.spec.single_frame_dim
    self.command_dim = policy.spec.command_dim
    self.velocity_dim = policy.spec.velocity_dim
    self.him_latent_dim = policy.spec.latent_dim
    self.use_him_estimator = policy.use_him_estimator
    self.encoder = (
      copy.deepcopy(policy.estimator.encoder)
      if self.use_him_estimator
      else nn.Identity()
    )
    self.mlp = copy.deepcopy(policy.mlp)
    self.deterministic_output = (
      policy.distribution.as_deterministic_output_module()
      if policy.distribution is not None
      else nn.Identity()
    )
    self.features = copy.deepcopy(policy.auxiliary.features)
    self.residual = copy.deepcopy(policy.residual)

  def forward(self, canonical_history: torch.Tensor) -> torch.Tensor:
    history = canonical_history
    if history.dim() != 2 or history.shape[-1] != self.history_dim:
      raise ValueError("Expected canonical [B, history_dim] proprioceptive history")
    if self.use_him_estimator:
      frames = history.reshape(-1, self.history_length, self.frame_dim)
      encoded = self.encoder(frames[..., self.command_dim :].flatten(1))
      velocity = encoded[:, : self.velocity_dim]
      latent = F.normalize(encoded[:, self.velocity_dim :], dim=-1, p=2.0)
      him_features = torch.cat((velocity, latent), dim=-1)
    else:
      him_features = history.new_zeros(
        (history.shape[0], self.velocity_dim + self.him_latent_dim)
      )
    output = self.mlp(torch.cat((history[:, : self.frame_dim], him_features), -1))
    if self.residual is not None:
      output = output + self.residual(self.features(history))
    return self.deterministic_output(output)

  @torch.jit.export
  def reset(self) -> None:
    pass
