"""小型合成rollout验证原生PPO真更新，不代表机器人训练或能力验收。"""

from pathlib import Path

import pytest
import torch
from model import Teacher, make_initial_modules
from ppo import build_ppo
from tensordict import TensorDict

SAMPLE = (
  Path(__file__).resolve().parents[1]
  / "black_reliability_20261119/evidence/WINDOWS_R800_SAMPLE.pt"
)


def setup_ppo():
  sample = torch.load(SAMPLE, map_location="cpu", weights_only=True)
  him, actor, _, _ = make_initial_modules(sample["learner"])
  teacher = Teacher(him, actor)
  row = sample["rows"][0]
  obs = TensorDict(
    {
      "history": row["history"][:4].reshape(4, 225).clone(),
      "geometry": torch.randn(4, 768),
      "critic": row["raw446"][:4].clone(),
    },
    batch_size=[4],
  )
  algorithm = build_ppo(
    obs,
    teacher,
    num_steps=4,
    ppo_options={
      "num_learning_epochs": 1,
      "num_mini_batches": 2,
      "schedule": "fixed",
      "learning_rate": 1e-4,
    },
  )
  return teacher, algorithm, obs


def test_native_ppo_reencodes_geometry_and_updates_only_allowed_parameters():
  torch.manual_seed(743)
  teacher, algorithm, obs = setup_ppo()
  assert algorithm.actor.mlp is teacher.actor
  assert algorithm.actor.policy is teacher
  optimizer_ids = {
    id(p) for group in algorithm.optimizer.param_groups for p in group["params"]
  }
  expected_ids = {
    id(p)
    for module in (algorithm.actor, algorithm.critic)
    for p in module.parameters()
    if p.requires_grad
  }
  assert optimizer_ids == expected_ids
  assert not optimizer_ids & {id(p) for p in teacher.him.parameters()}
  assert not algorithm.optimizer.state

  before = {name: p.detach().clone() for name, p in teacher.named_parameters()}
  critic_before = {
    name: p.detach().clone() for name, p in algorithm.critic.named_parameters()
  }
  gradient_modes = []
  hook = teacher.terrain_encoder.register_forward_pre_hook(
    lambda module, args: gradient_modes.append(torch.is_grad_enabled())
  )
  with torch.no_grad():
    for tick in range(4):
      actions = algorithm.act(obs)
      rewards = -actions.square().mean(dim=-1) + torch.arange(4) * 0.1
      next_obs = obs.clone()
      next_obs["history"] = obs["history"] + 0.001
      next_obs["geometry"] = obs["geometry"] + (tick + 1) * 0.002
      algorithm.process_env_step(
        next_obs, rewards, torch.zeros(4, dtype=torch.bool), {}
      )
      obs = next_obs
    algorithm.compute_returns(obs)

  assert algorithm.storage.observations["geometry"].shape == (4, 4, 768)
  losses = algorithm.update()
  hook.remove()
  assert gradient_modes == [False] * 4 + [True] * 2
  assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
  assert algorithm.optimizer.state
  # 原生PPO在update末尾更新SplitNormalizer，统计随新模型重新建立。
  normalizer = algorithm.critic.obs_normalizer.state_dict()
  assert normalizer["raw72.count"].item() == 16
  assert normalizer["terrain.count"].item() == 16
  after = dict(teacher.named_parameters())
  for prefix in ("actor.", "terrain_encoder."):
    assert any(
      not torch.equal(before[name], after[name])
      for name in before
      if name.startswith(prefix)
    )
  assert all(
    torch.equal(before[name], after[name]) for name in before if name.startswith("him.")
  )
  assert any(
    not torch.equal(value, dict(algorithm.critic.named_parameters())[name])
    for name, value in critic_before.items()
  )


def test_actor_cannot_read_critic_and_wrapper_preserves_initial_head():
  teacher, algorithm, obs = setup_ppo()
  expected = teacher(obs["history"].reshape(-1, 5, 45), obs["geometry"])
  assert torch.equal(algorithm.actor(obs), expected)
  altered = obs.clone()
  altered["critic"] = torch.randn_like(obs["critic"]) * 1000
  assert torch.equal(algorithm.actor(altered), expected)
  teacher.freeze()
  with pytest.raises(ValueError, match="trainable"):
    build_ppo(obs, teacher, num_steps=4)


def test_native_timeout_approximation_never_bootstraps_true_terminal():
  _, algorithm, obs = setup_ppo()
  with torch.no_grad():
    algorithm.act(obs)
    assert algorithm.transition.values is not None
    values = algorithm.transition.values.clone().squeeze(1)
    reset_obs = obs.clone()
    reset_obs["critic"] += 100
    # 纯超时沿用RSL的V(s_t)近似；真实终止和普通步不加该项。
    mask = torch.tensor([True, False, False, False])
    algorithm.process_env_step(
      reset_obs,
      torch.ones(4),
      torch.tensor([1, 1, 1, 0]),
      {"time_outs": mask},
    )
  expected = torch.ones(4) + algorithm.gamma * values * mask
  torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], expected)
