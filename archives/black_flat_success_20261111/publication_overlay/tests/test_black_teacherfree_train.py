"""Diagnostics leave one PPO actor/critic and correct reward-step alignment."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.black_teacherfree import TaskLoggedPPO
from mjlab.scripts.train_black_teacherfree import ProfileLoggedWrapper, make_configs
from mjlab.tasks.velocity.config.black_teacher.teacherfree_command import (
  TeacherFreeVelocityCommand,
  TeacherFreeVelocityCommandCfg,
)


def algorithm(cls):
  torch.manual_seed(19)
  obs = TensorDict({"actor": torch.randn(6, 72)}, batch_size=[6])
  groups = {"actor": ["actor"], "critic": ["actor"]}
  actor = MLPModel(
    obs,
    groups,
    "actor",
    12,
    hidden_dims=[16],
    distribution_cfg={"class_name": "GaussianDistribution"},
  )
  critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16])
  storage = RolloutStorage("rl", 6, 3, obs, [12], "cpu")
  alg = cls(
    actor,
    critic,
    storage,
    num_learning_epochs=2,
    num_mini_batches=2,
    learning_rate=5e-5,
    schedule="fixed",
    device="cpu",
  )
  if isinstance(alg, TaskLoggedPPO):
    alg.profile_id = torch.tensor([0, 1, 2, 0, 1, 2])
  torch.manual_seed(27)
  with torch.no_grad():
    for _ in range(3):
      alg.act(obs)
      obs = TensorDict({"actor": torch.randn(6, 72)}, batch_size=[6])
      alg.process_env_step(obs, torch.randn(6), torch.zeros(6).bool(), {})
    alg.compute_returns(obs)
  return alg


def test_logged_algorithm_exactly_preserves_upstream_ppo():
  original, logged = algorithm(PPO), algorithm(TaskLoggedPPO)
  torch.manual_seed(41)
  expected = original.update()
  torch.manual_seed(41)
  actual = logged.update()
  for k in expected:
    assert actual[k] == pytest.approx(expected[k], abs=1e-8)
  for a, b in ((original.actor, logged.actor), (original.critic, logged.critic)):
    for key, value in a.state_dict().items():
      torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
  assert set(logged.save()) == {
    "actor_state_dict",
    "critic_state_dict",
    "optimizer_state_dict",
  }
  assert not hasattr(logged, "teachers") and not hasattr(logged, "dagger")
  for name in ("low", "omni", "high"):
    assert f"{name}_advantage_std" in actual


def test_one_unchanged_actor_critic_and_explicit_reward_config():
  cfg, rl = make_configs(4096, 51001)
  assert set(cfg.observations) == {"actor", "critic"}
  assert (
    cast(TeacherFreeVelocityCommandCfg, cfg.commands["twist"]).profile_seed == 51001
  )
  assert rl.actor.hidden_dims == (512, 256, 128)
  assert rl.critic.hidden_dims == (512, 256, 128)
  assert rl.algorithm.schedule == "fixed" and rl.algorithm.learning_rate == 5e-5
  assert cfg.rewards["teacherfree_linear_huber"].weight == -0.2
  assert cfg.rewards["teacherfree_yaw_huber"].weight == -0.1


def test_reward_log_uses_applied_command_before_resample(monkeypatch):
  command = TeacherFreeVelocityCommand.__new__(TeacherFreeVelocityCommand)
  command.vel_command_b = torch.tensor(
    [[0.1, 0.0, 0.0], [0.7, 0.0, 0.0], [1.0, 0.0, 0.0]]
  )
  command.profile_id = torch.tensor([0, 1, 2])
  rates = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: command),
    reward_manager=SimpleNamespace(_step_reward=rates, active_terms=["a", "b"]),
    step_dt=0.02,
    common_step_counter=100,
    reset_terminated=torch.tensor([False, False, True]),
  )
  env.unwrapped = env
  wrapper = ProfileLoggedWrapper.__new__(ProfileLoggedWrapper)
  wrapper.env = cast(ManagerBasedRlEnv, env)

  def step(self, actions):
    command.vel_command_b.zero_()  # reset/resample must not change previous reward labels
    return ({}, rates.sum(-1) * 0.02, torch.zeros(3), {"log": {}})

  monkeypatch.setattr(RslRlVecEnvWrapper, "step", step)
  out = wrapper.step(torch.zeros(3, 12))
  logs = out[3]["log"]
  assert logs["TaskReward/high/high_gate"] == 1
  assert logs["TaskReward/low/low_gate"] == 1
  assert logs["TaskReward/high/total_rate"] == 11
  assert logs["TaskReward/high/terminated_fraction"] == 1
