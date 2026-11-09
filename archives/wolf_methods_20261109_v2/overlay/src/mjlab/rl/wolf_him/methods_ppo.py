"""Actual PPO integration for optional terrain distillation and prediction.

Auxiliary encoders stay fixed for all PPO epochs and update once afterwards.
Only a bounded sample of the on-policy rollout is used, with no replay buffer.
Terminal targets reuse HIM's pre-reset recorder; no reset-to-next-episode pairs.
"""

from __future__ import annotations

import math
from typing import cast

import torch

from .dual_critic import DualRewardPPOMixin
from .methods_targets import validated_terminal_extras
from .perception_models import PerceptionHIMPolicy
from .ppo import HIMPPO
from .spec import canonical_history
from .storage import HIMRolloutStorage


class MethodsPPO(DualRewardPPOMixin, HIMPPO):
  storage: HIMRolloutStorage

  @property
  def him_actor(self) -> PerceptionHIMPolicy:
    return cast(PerceptionHIMPolicy, self.actor)

  def __init__(
    self,
    *args,
    auxiliary_learning_rate: float = 3e-4,
    auxiliary_batch_size: int = 256,
    auxiliary_max_samples: int = 2048,
    distillation_loss_weight: float = 1.0,
    prediction_loss_weight: float = 1.0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if auxiliary_batch_size <= 0 or auxiliary_max_samples <= 0:
      raise ValueError("Auxiliary sample and batch sizes must be positive")
    if not math.isfinite(auxiliary_learning_rate) or auxiliary_learning_rate <= 0:
      raise ValueError("Auxiliary learning rate must be positive")
    if not all(
      math.isfinite(w) and w >= 0
      for w in (distillation_loss_weight, prediction_loss_weight)
    ):
      raise ValueError("Auxiliary loss weights must be nonnegative")
    self.auxiliary_batch_size = auxiliary_batch_size
    self.auxiliary_max_samples = auxiliary_max_samples
    self.auxiliary_weights = {
      "terrain_reconstruction": distillation_loss_weight,
      "terrain_distillation": distillation_loss_weight,
      "next_proprioception": prediction_loss_weight,
    }
    self.auxiliary_optimizer = (
      torch.optim.Adam(
        self.him_actor.auxiliary.parameters(), lr=auxiliary_learning_rate
      )
      if self.him_actor.methods_enabled
      else None
    )

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    extras = validated_terminal_extras(dones, extras, self.interface, self.spec)
    super().process_env_step(obs, rewards, dones, extras)

  def _update_estimator(self, batch, lr: float) -> tuple[float, float]:
    if not self.him_actor.use_him_estimator:
      return 0.0, 0.0
    return super()._update_estimator(batch, lr)

  def update(self) -> dict[str, float]:
    # Capture a bounded sample before HIMPPO clears the rollout cursor. Do not
    # retain a second full rollout or update encoders midway through PPO epochs.
    sampled = None
    if self.auxiliary_optimizer is not None:
      if self.storage.step != self.storage.num_transitions_per_env:
        raise ValueError("Methods update requires a complete on-policy rollout")
      size = self.storage.step * self.storage.num_envs
      count = min(size, self.auxiliary_max_samples)
      indices = torch.randperm(size, device=self.device)[:count]
      flat_observations = self.storage.observations.flatten(0, 1)
      assert flat_observations is not None
      sampled = (
        flat_observations[indices],
        self.storage.actions.flatten(0, 1)[indices],
        self.storage.next_estimator_input.flatten(0, 1)[indices],
      )
    losses = super().update()
    if sampled is not None:
      losses.update(self._update_auxiliary(*sampled))
    return losses

  def _update_auxiliary(self, observations, actions, next_inputs) -> dict[str, float]:
    optimizer = self.auxiliary_optimizer
    assert optimizer is not None
    totals: dict[str, float] = {}
    size = actions.shape[0]
    aux = self.him_actor.auxiliary
    for start in range(0, size, self.auxiliary_batch_size):
      stop = min(size, start + self.auxiliary_batch_size)
      batch = observations[start:stop]
      losses = aux.losses(
        canonical_history(batch[self.interface.history_group]),
        actions[start:stop],
        next_inputs[start:stop],
        batch[self.him_actor.teacher_local_group] if aux.distillation else None,
        batch[self.him_actor.teacher_global_group] if aux.distillation else None,
      )
      objective = torch.stack(
        [self.auxiliary_weights[key] * value for key, value in losses.items()]
      ).sum()
      if not torch.isfinite(objective):
        raise FloatingPointError("Nonfinite methods auxiliary loss")
      optimizer.zero_grad(set_to_none=True)
      objective.backward()
      torch.nn.utils.clip_grad_norm_(
        aux.parameters(), self.max_grad_norm, error_if_nonfinite=True
      )
      optimizer.step()
      for key, value in losses.items():
        totals[key] = totals.get(key, 0.0) + value.item() * (stop - start) / size
    return {f"methods/{key}": value for key, value in totals.items()}

  def _methods_signature(self) -> dict:
    aux = self.him_actor.auxiliary
    return {
      "version": 2,
      "use_him_estimator": self.him_actor.use_him_estimator,
      "distillation": aux.distillation,
      "prediction": aux.prediction,
      "latent_dim": aux.latent_dim,
      "prediction_target_dim": aux.target_dim,
      "history_dim": self.spec.history_dim,
      "action_dim": self.spec.action_dim,
      "teacher_local_rays": aux.teacher_local.num_rays if aux.distillation else 0,
      "teacher_global_rays": aux.teacher_global.num_rays if aux.distillation else 0,
      "loss_weights": self.auxiliary_weights,
    }

  def save(self) -> dict:
    checkpoint = super().save()
    checkpoint["methods_signature"] = self._methods_signature()
    if self.auxiliary_optimizer is not None:
      checkpoint["methods_optimizer_state_dict"] = self.auxiliary_optimizer.state_dict()
    return checkpoint

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    # Check BEFORE loading weights: never silently resume an incompatible method.
    if loaded_dict.get("methods_signature") != self._methods_signature():
      raise ValueError("Methods checkpoint configuration differs; use a matching run")
    load_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
    if self.auxiliary_optimizer is not None and load_optimizer:
      if "methods_optimizer_state_dict" not in loaded_dict:
        raise ValueError("Methods resume checkpoint is missing auxiliary optimizer")
    result = super().load(loaded_dict, load_cfg, strict)
    if self.auxiliary_optimizer is not None and load_optimizer:
      self.auxiliary_optimizer.load_state_dict(
        loaded_dict["methods_optimizer_state_dict"]
      )
    return result
