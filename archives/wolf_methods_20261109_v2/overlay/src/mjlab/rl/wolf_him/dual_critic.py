"""Independent locomotion/placement values with separately normalized GAE.

This is a BeamDojo-inspired PPO adapter, not TD3's minimum-of-two-Q update.
It deliberately preserves scalar actor advantages and the existing HIM update.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable
from torch import nn


class DualCriticModel(nn.Module):
  """Two independent deterministic models with a shared observation contract."""

  is_recurrent = False

  def __init__(
    self,
    obs,
    obs_groups,
    obs_set,
    output_dim=1,
    *,
    base_critic_class="MLPModel",
    base_critic_kwargs=None,
    **model_kwargs,
  ):
    super().__init__()
    if output_dim not in (1, 2):
      raise ValueError("Dual critic expects scalar values per reward channel")
    options = {**model_kwargs, **(base_critic_kwargs or {})}
    if options.get("distribution_cfg") is not None:
      raise ValueError("Value models must be deterministic")
    critic_cls = resolve_callable(base_critic_class)
    self.models = nn.ModuleList(
      [critic_cls(obs, obs_groups, obs_set, 1, **options) for _ in range(2)]
    )
    if any(model.is_recurrent for model in self.models):
      raise ValueError("DualCriticModel currently supports feedforward critics only")

  def forward(self, obs, **kwargs):
    return torch.cat([model(obs, **kwargs) for model in self.models], dim=-1)

  def get_hidden_state(self):
    return None

  def reset(self, dones=None, hidden_state=None):
    for model in self.models:
      cast(MLPModel, model).reset(dones, hidden_state=hidden_state)

  def update_normalization(self, obs):
    for model in self.models:
      cast(MLPModel, model).update_normalization(obs)


def separate_gae(rewards, values, dones, last_values, gamma, lam, weights):
  """Compute two returns and normalize each advantage before weighted summation."""
  if rewards.shape != values.shape or values.shape[-1] != 2:
    raise ValueError("Expected matching [time, env, 2] rewards and values")
  if len(weights) != 2 or not all(math.isfinite(w) for w in weights):
    raise ValueError("Expected two finite advantage weights")
  if dones.shape != (*values.shape[:2], 1):
    raise ValueError("dones must have shape [time, env, 1]")
  if last_values.shape != values.shape[1:]:
    raise ValueError("last_values must have shape [env, 2]")
  advantages = torch.zeros_like(values)
  running = torch.zeros_like(last_values)
  for step in reversed(range(values.shape[0])):
    next_value = last_values if step == values.shape[0] - 1 else values[step + 1]
    alive = 1.0 - dones[step].float()
    delta = rewards[step] + gamma * alive * next_value - values[step]
    running = delta + gamma * lam * alive * running
    advantages[step] = running
  returns = advantages + values
  mean = advantages.mean(dim=(0, 1), keepdim=True)
  std = advantages.std(dim=(0, 1), keepdim=True, unbiased=False)
  normalized = (advantages - mean) / std.clamp_min(1e-8)
  combined = (normalized * values.new_tensor(weights)).sum(-1, keepdim=True)
  return returns, combined, normalized


class DualRewardPPOMixin(PPO):
  """Place before HIMPPO/PPO in the MRO; native minibatches retain two values.

  The placement reward must already be included in the environment total and
  be published separately in extras, with exactly the same weight/dt scaling.
  Time-limit bootstrapping is applied independently to both value channels.
  """

  def __init__(
    self,
    *args,
    dual_critic_enabled=True,
    sparse_reward_key="wolf_sparse_reward",
    advantage_weights=(1.0, 1.0),
    **kwargs,
  ):
    self.dual_critic_enabled = dual_critic_enabled
    self.sparse_reward_key = sparse_reward_key
    self.advantage_weights = tuple(advantage_weights)
    if len(self.advantage_weights) != 2 or not all(
      math.isfinite(w) for w in self.advantage_weights
    ):
      raise ValueError("Expected two finite advantage weights")
    super().__init__(*args, **kwargs)
    if not dual_critic_enabled:
      if isinstance(self._raw_critic, DualCriticModel):
        raise ValueError("Disabled dual critic requires a single-value model")
      return
    if not isinstance(self._raw_critic, DualCriticModel):
      raise ValueError("Enabled dual critic requires DualCriticModel")
    if self.rnd or self.normalize_advantage_per_mini_batch:
      raise ValueError(
        "Dual reward PPO does not support RND or minibatch normalization"
      )
    if self.is_multi_gpu:
      raise ValueError("Dual reward GAE currently supports single-device training only")
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise ValueError("Dual reward PPO currently requires feedforward models")
    st = self.storage
    if st.step:
      raise ValueError("Dual reward buffers must be configured before rollout")
    shape = (st.num_transitions_per_env, st.num_envs, 2)
    self.channel_rewards = torch.zeros(shape, device=st.device)
    st.values = torch.zeros_like(self.channel_rewards)
    st.returns = torch.zeros_like(self.channel_rewards)

  def process_env_step(self, obs, rewards, dones, extras):
    if not self.dual_critic_enabled:
      return super().process_env_step(obs, rewards, dones, extras)
    if self.sparse_reward_key not in extras:
      raise KeyError(f"Missing reward channel {self.sparse_reward_key!r}")
    sparse = extras[self.sparse_reward_key].to(rewards.device).reshape_as(rewards)
    if not torch.isfinite(sparse).all():
      raise ValueError("Sparse reward channel contains non-finite values")
    channels = torch.stack((rewards - sparse, sparse), dim=-1)
    if "time_outs" in extras:
      assert self.transition.values is not None
      channels = channels + (
        self.gamma
        * self.transition.values
        * extras["time_outs"].to(channels.device).reshape(-1, 1)
      )
    self.channel_rewards[self.storage.step].copy_(channels)
    # Parent still records HIM successor labels and owns transition/reset logic.
    # Its scalar reward buffer is unused by our GAE; avoid scalar timeout code.
    parent_extras = {k: v for k, v in extras.items() if k != "time_outs"}
    super().process_env_step(obs, rewards, dones, parent_extras)

  def compute_returns(self, obs):
    if not self.dual_critic_enabled:
      return super().compute_returns(obs)
    if self.storage.step != self.storage.num_transitions_per_env:
      raise ValueError("Compute returns only after a complete rollout")
    with torch.no_grad():
      last_values = self.critic(obs)
      returns, advantages, _ = separate_gae(
        self.channel_rewards,
        self.storage.values,
        self.storage.dones,
        last_values,
        self.gamma,
        self.lam,
        self.advantage_weights,
      )
    self.storage.returns.copy_(returns)
    self.storage.advantages = advantages

  def save(self):
    result = super().save()
    result["dual_reward_contract"] = {
      "version": 1,
      "enabled": self.dual_critic_enabled,
      "channels": ("locomotion", self.sparse_reward_key),
      "advantage_weights": self.advantage_weights,
    }
    return result

  def load(self, loaded_dict, load_cfg, strict):
    contract = loaded_dict.get("dual_reward_contract")
    expected = {
      "version": 1,
      "enabled": self.dual_critic_enabled,
      "channels": ("locomotion", self.sparse_reward_key),
      "advantage_weights": self.advantage_weights,
    }
    full_resume = load_cfg is None or load_cfg.get("optimizer", False)
    if full_resume and contract != expected:
      raise ValueError(
        "Dual reward checkpoint contract mismatch; use explicit warm start"
      )
    return super().load(loaded_dict, load_cfg, strict)
