"""Selective PPO continuation with student-only rollouts and bounded DAgger.

The original 72-D policy stays intact. No HIM model, history, or training is used.
"""

from copy import deepcopy

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F


def teacher_selection(command, target, healthy, age):
  """Disjoint conservative domains. IDs: 0 none, 1 low, 2 Omni; high disabled."""

  def low_domain(c):
    return ((c.abs() > 1e-5).sum(dim=1) <= 1) & (c.abs().max(dim=1).values <= 0.30)

  valid = (
    healthy
    & (age >= 1.0)
    & torch.isfinite(command).all(dim=1)
    & torch.isfinite(target).all(dim=1)
  )
  low = low_domain(command) & low_domain(target) & valid
  # Never distill the known unreliable high mixed boundaries.
  pure = (command.abs() > 0.02).sum(dim=1) <= 1
  moderate = (command[:, :2].norm(dim=1) <= 0.85) & (command[:, 2].abs() <= 0.35)
  target_ok = (target.abs() > 0.02).sum(dim=1) <= 1
  target_ok |= (target[:, :2].norm(dim=1) <= 0.85) & (target[:, 2].abs() <= 0.35)
  omni = valid & ~low & (pure | moderate) & target_ok
  return low.long() + 2 * omni.long()


def masked_huber(prediction, label, teacher_id, weights, delta=0.5):
  per_sample = F.huber_loss(
    prediction, label.detach(), delta=delta, reduction="none"
  ).mean(-1)
  result = prediction.sum() * 0.0
  for index, weight in enumerate(weights, start=1):
    selected = teacher_id == index
    if selected.any():
      result = result + weight * per_sample[selected].mean()
  return result


class RecentDAgger:
  """Bounded recent-window aggregation, not a full-history reservoir."""

  def __init__(self, capacity=100000):
    self.capacity = capacity
    self.position = 0
    self.size = 0
    self.total_seen = 0
    self.obs = torch.empty(capacity, 72)
    self.action = torch.empty(capacity, 12)
    self.teacher = torch.empty(capacity, dtype=torch.long)
    self.generator = torch.Generator().manual_seed(torch.initial_seed())

  def add(self, obs, action, teacher):
    selected = teacher > 0
    obs, action, teacher = (
      obs[selected].detach().cpu(),
      action[selected].detach().cpu(),
      teacher[selected].cpu(),
    )
    count = len(obs)
    self.total_seen += count
    if count > self.capacity:
      obs, action, teacher = (
        obs[-self.capacity :],
        action[-self.capacity :],
        teacher[-self.capacity :],
      )
      count = self.capacity
    if not count:
      return
    indices = (torch.arange(count) + self.position) % self.capacity
    self.obs[indices], self.action[indices], self.teacher[indices] = (
      obs,
      action,
      teacher,
    )
    self.position = (self.position + count) % self.capacity
    self.size = min(self.capacity, self.size + count)

  def sample(self, count, device):
    ids = torch.randint(self.size, (count,), generator=self.generator)
    return (
      self.obs[ids].to(device),
      self.action[ids].to(device),
      self.teacher[ids].to(device),
    )

  def state_dict(self):
    return dict(
      capacity=self.capacity,
      position=self.position,
      size=self.size,
      total_seen=self.total_seen,
      obs=self.obs[: self.size],
      action=self.action[: self.size],
      teacher=self.teacher[: self.size],
      rng=self.generator.get_state(),
    )

  def load_state_dict(self, state):
    self.__init__(state["capacity"])
    self.position, self.size, self.total_seen = (
      state["position"],
      state["size"],
      state["total_seen"],
    )
    for key in ("obs", "action", "teacher"):
      getattr(self, key)[: self.size].copy_(state[key].cpu())
    self.generator.set_state(state["rng"].cpu())


class SelectiveTeacherPPO(PPO):
  """RSL 5.5.1 feed-forward PPO plus gradient-capped Huber per PPO step."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.rnd or self.symmetry or self.is_multi_gpu or self.actor.is_recurrent:
      raise ValueError("Continuation supports plain single-GPU MLP PPO only")
    self.teachers = []
    self.sim_env = None
    self.stage = 1
    self.stage_update = 0
    self.dagger = RecentDAgger()
    self.pending = []
    self.coverage = torch.zeros(3, device=self.device)
    self.lr_cap = 1e-4
    self.aux_enabled = True

  def bind(self, env, low_checkpoint, omni_checkpoint, stage):
    self.sim_env, self.stage = env, stage
    self.teachers = []
    for checkpoint in (low_checkpoint, omni_checkpoint):
      teacher = deepcopy(self.actor)
      data = torch.load(checkpoint, map_location=self.device, weights_only=False)
      teacher.load_state_dict(data["actor_state_dict"], strict=True)
      teacher.eval().requires_grad_(False)
      self.teachers.append(teacher)

  def act(self, obs):
    if self.sim_env is not None:
      env = self.sim_env
      term = env.command_manager.get_term("twist")
      robot = env.scene["robot"]
      healthy = robot.data.projected_gravity_b[:, 2] < -0.866
      healthy &= robot.data.root_link_pos_w[:, 2] > 0.20
      healthy &= env.episode_length_buf * env.step_dt >= 0.5
      selected = teacher_selection(
        term.command, term.target_command, healthy, term.transition_age
      )
      labels = torch.zeros(env.num_envs, 12, device=self.device)
      for index, teacher in enumerate(self.teachers, start=1):
        mask = selected == index
        if mask.any():
          labels[mask] = teacher(obs[mask])
      selected[~torch.isfinite(labels).all(dim=1)] = 0
      self.pending.append((obs["actor"].detach().clone(), labels, selected))
      self.coverage += torch.bincount(selected, minlength=3)
    # These are always the student's sampled actions. Teacher labels never act.
    return super().act(obs)

  def auxiliary_weights(self):
    if self.stage == 1:
      low = 0.5 * min(1.0, (self.stage_update + 1) / 50)
      if self.stage_update >= 50:
        low = max(0.2, 0.5 - 0.3 * (self.stage_update - 50) / 9950)
      return (low, 0.10)
    return (0.15, 0.15)

  def update(self):
    for obs, label, teacher in self.pending:
      self.dagger.add(obs, label, teacher)
    self.pending.clear()
    mean = dict(
      value=0.0, surrogate=0.0, entropy=0.0, selective_huber=0.0, auxiliary_scale=0.0
    )
    actor_params = tuple(self.actor.parameters())
    count = 0
    for batch in self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    ):
      assert batch.advantages is not None
      assert batch.actions is not None
      assert batch.old_distribution_params is not None
      assert batch.old_actions_log_prob is not None
      assert batch.values is not None
      assert batch.returns is not None
      if self.normalize_advantage_per_mini_batch:
        batch.advantages = (batch.advantages - batch.advantages.mean()) / (
          batch.advantages.std() + 1e-8
        )
      self.actor(batch.observations, stochastic_output=True)
      log_prob = self.actor.get_output_log_prob(batch.actions)
      entropy = self.actor.output_entropy
      values = self.critic(batch.observations)
      if self.schedule == "adaptive" and self.desired_kl is not None:
        with torch.no_grad():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, self.actor.output_distribution_params
          ).mean()
          if kl > self.desired_kl * 2:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif 0 < kl < self.desired_kl / 2:
            self.learning_rate = min(self.lr_cap, self.learning_rate * 1.5)
          self.learning_rate = min(self.lr_cap, self.learning_rate)
          for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
      ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze())
      assert batch.advantages is not None
      advantages = batch.advantages.squeeze()
      surrogate = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
      ).mean()
      if self.use_clipped_value_loss:
        clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value = torch.maximum(
          (values - batch.returns).square(), (clipped - batch.returns).square()
        ).mean()
      else:
        value = (values - batch.returns).square().mean()
      loss = (
        surrogate + self.value_loss_coef * value - self.entropy_coef * entropy.mean()
      )
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      aux_value, scale_value = 0.0, 0.0
      if self.aux_enabled and self.dagger.size:
        observations, labels, teacher_ids = self.dagger.sample(
          min(4096, self.dagger.size), self.device
        )
        prediction = self.actor(
          TensorDict({"actor": observations}, batch_size=[len(observations)])
        )
        aux = masked_huber(prediction, labels, teacher_ids, self.auxiliary_weights())
        if not torch.isfinite(aux):
          raise FloatingPointError("Nonfinite selective Huber")
        gradients = torch.autograd.grad(aux, actor_params, allow_unused=True)
        ppo_norm = torch.sqrt(
          torch.stack(
            [p.grad.square().sum() for p in actor_params if p.grad is not None]
          ).sum()
        )
        aux_norm = torch.sqrt(
          torch.stack([g.square().sum() for g in gradients if g is not None]).sum()
        )
        scale = (0.30 * ppo_norm / (aux_norm + 1e-12)).clamp(max=1.0)
        for param, grad in zip(actor_params, gradients, strict=True):
          if grad is not None:
            if param.grad is None:
              param.grad = scale * grad
            else:
              param.grad.add_(scale * grad)
        aux_value, scale_value = aux.item(), scale.item()
      if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite PPO loss")
      nn.utils.clip_grad_norm_(
        actor_params, self.max_grad_norm, error_if_nonfinite=True
      )
      nn.utils.clip_grad_norm_(
        self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True
      )
      self.optimizer.step()
      for key, val in dict(
        value=value.item(),
        surrogate=surrogate.item(),
        entropy=entropy.mean().item(),
        selective_huber=aux_value,
        auxiliary_scale=scale_value,
      ).items():
        mean[key] += val
      count += 1
    all_obs = self.storage.observations.flatten(0, 1)
    assert isinstance(all_obs, TensorDict)
    self.actor.update_normalization(all_obs)
    self.critic.update_normalization(all_obs)
    self.storage.clear()
    mean = {key: val / max(1, count) for key, val in mean.items()}
    total = self.coverage.sum().clamp_min(1)
    for i, name in enumerate(("none", "low", "omni")):
      mean[f"teacher_{name}_fraction"] = (self.coverage[i] / total).item()
    self.coverage.zero_()
    mean.update(dagger_size=self.dagger.size, dagger_total_seen=self.dagger.total_seen)
    self.stage_update += 1
    return mean

  def save(self):
    return {
      **super().save(),
      "continuation": dict(
        stage=self.stage,
        stage_update=self.stage_update,
        dagger=self.dagger.state_dict(),
      ),
    }

  def load(self, loaded_dict, load_cfg, strict):
    iteration = super().load(loaded_dict, load_cfg, strict)
    if "continuation" in loaded_dict:
      state = loaded_dict["continuation"]
      self.stage, self.stage_update = state["stage"], state["stage_update"]
      self.dagger.load_state_dict(state["dagger"])
    self.learning_rate = min(self.lr_cap, self.learning_rate)
    for group in self.optimizer.param_groups:
      group["lr"] = self.learning_rate
    return iteration
