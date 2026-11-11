"""CPU-only contracts for applied-command-conditioned teacher-free rewards."""

from copy import deepcopy
from types import SimpleNamespace as NS
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardManager
from mjlab.tasks.velocity.config.black_teacher import teacherfree_rewards as r
from mjlab.tasks.velocity.config.black_teacher.env_cfgs import (
  black_teacher_omni_env_cfg,
)


def fake_env(commands) -> ManagerBasedRlEnv:
  command = torch.tensor(commands, dtype=torch.float64)
  data = NS(
    root_link_lin_vel_b=torch.zeros_like(command),
    root_link_ang_vel_b=torch.zeros_like(command),
  )
  return cast(
    ManagerBasedRlEnv,
    NS(
      command_manager=NS(get_command=lambda _: command),
      scene={"robot": NS(data=data)},
      step_dt=0.02,
      num_envs=len(commands),
      device="cpu",
    ),
  )


def matched(env):
  command = env.command_manager.get_command("twist")
  env.scene["robot"].data.root_link_lin_vel_b[:, :2] = command[:, :2]
  env.scene["robot"].data.root_link_ang_vel_b[:, 2] = command[:, 2]


def test_matched_maximum_two_each_and_no_huber_cost():
  env = fake_env([[0, 0, 0], [0.1, -0.1, 0.08], [0.5, 0, 0.3], [2, -1, 1]])
  matched(env)
  for fn in (r.track_linear_velocity, r.track_angular_velocity):
    torch.testing.assert_close(2 * fn(env, "twist"), torch.full((4,), 2.0).double())
  for fn in (r.linear_velocity_huber_cost, r.yaw_velocity_huber_cost):
    assert fn(env, "twist").count_nonzero() == 0


def test_low_standing_vs_following_numeric():
  env = fake_env([[0.1, 0, 0.08]])
  assert (2 * r.track_linear_velocity(env, "twist")).item() == pytest.approx(
    1.3286688803
  )
  assert (2 * r.track_angular_velocity(env, "twist")).item() == pytest.approx(
    1.3551610128
  )
  matched(env)
  assert r.track_linear_velocity(env, "twist").item() == 1
  assert r.track_angular_velocity(env, "twist").item() == 1


@pytest.mark.parametrize(
  "axis, low_bounds, high_bounds",
  [
    (0, (0.45, 0.60), (0.5, 1.0)),
    (2, (0.27, 0.36), (0.25, 0.5)),
  ],
)
def test_gate_boundaries_and_finite_gradients(axis, low_bounds, high_bounds):
  values = [0, *low_bounds, *high_bounds]
  commands = torch.zeros(len(values), 3, dtype=torch.float64)
  commands[:, axis] = torch.tensor(values)
  commands.requires_grad_()
  low, high = r.command_gates(commands)
  torch.testing.assert_close(low[:3], torch.tensor([1.0, 1.0, 0.0]).double())
  torch.testing.assert_close(high[[0, 3, 4]], torch.tensor([0.0, 0.0, 1.0]).double())
  (low + high).sum().backward()
  assert commands.grad is not None
  assert torch.isfinite(commands.grad).all()
  for gate_idx, bounds in enumerate((low_bounds, high_bounds)):
    for bound in bounds:
      x = torch.zeros(3, 3, dtype=torch.float64)
      x[:, axis] = torch.tensor([bound - 1e-7, bound, bound + 1e-7])
      y = r.command_gates(x)[gate_idx]
      assert (y.max() - y.min()).item() < 1e-10
  x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], requires_grad=True)
  r.smoothstep(x).sum().backward()
  torch.testing.assert_close(x.grad, torch.tensor([0.0, 0.0, 1.5, 0.0, 0.0]))


def test_pure_yaw_no_axis_leak_and_original_off_axis_penalties():
  env = fake_env([[0, 0, 0.5]])
  assert r.track_linear_velocity(env, "twist").item() == 1
  assert r.linear_velocity_huber_cost(env, "twist").item() == 0
  env.scene["robot"].data.root_link_lin_vel_b[:, 2] = 0.4
  env.scene["robot"].data.root_link_ang_vel_b[:, :2] = 0.2
  assert r.linear_velocity_huber_cost(env, "twist").item() == 0
  assert r.yaw_velocity_huber_cost(env, "twist").item() == pytest.approx(0.09375)
  for command in ([0, 0, 0], [2, 0, 1]):
    env = fake_env([command])
    matched(env)
    env.scene["robot"].data.root_link_lin_vel_b[:, 2] = 0.4
    env.scene["robot"].data.root_link_ang_vel_b[:, :2] = 0.2
    torch.testing.assert_close(
      r.track_linear_velocity(env, "twist"),
      r.base.track_linear_velocity(env, 0.5, "twist"),
    )
    torch.testing.assert_close(
      r.track_angular_velocity(env, "twist"),
      r.base.track_angular_velocity(env, 0.5**0.5, "twist"),
    )


def test_applied_not_target_or_profile_and_not_dt():
  env = fake_env([[0.1, 0, 0.08]])
  functions = (
    r.track_linear_velocity,
    r.track_angular_velocity,
    r.linear_velocity_huber_cost,
    r.yaw_velocity_huber_cost,
  )
  before = [fn(env, "twist") for fn in functions]
  cast(NS, env.command_manager).profile_id = torch.tensor([2])
  cast(NS, env.command_manager).target_command = torch.tensor([[3.0, 2.0, 1.0]])
  cast(NS, env).step_dt = 0.5
  for fn, expected in zip(functions, before, strict=True):
    torch.testing.assert_close(fn(env, "twist"), expected)
  cast(torch.Tensor, env.command_manager.get_command("twist"))[:] = torch.tensor(
    [2.0, 0.0, 1.0]
  )
  assert r.linear_velocity_huber_cost(env, "twist").item() > 0


def test_huber_low_high_error_scales_and_gradient():
  e = torch.tensor([0.0, 0.1, 0.5, 1.0, -2.0], requires_grad=True)
  torch.testing.assert_close(
    r.huber(e, 0.5), torch.tensor([0.0, 0.005, 0.125, 0.375, 0.875])
  )
  r.huber(e, 0.5).sum().backward()
  torch.testing.assert_close(e.grad, torch.tensor([0.0, 0.1, 0.5, 0.5, -0.5]))
  with pytest.raises(ValueError):
    r.huber(e, 0)
  env = fake_env([[0.1, 0, 0.08], [1, -1, 0.5], [2, 0, 1]])
  torch.testing.assert_close(
    r.linear_velocity_huber_cost(env, "twist"),
    torch.tensor([0.0, 0.75, 0.875]).double(),
  )
  torch.testing.assert_close(
    r.yaw_velocity_huber_cost(env, "twist"),
    torch.tensor([0.0, 0.09375, 0.21875]).double(),
  )
  assert torch.isfinite(r.track_linear_velocity(env, "twist")).all()


def test_config_only_four_intended_terms_common_rewards_unchanged():
  cfg = black_teacher_omni_env_cfg()
  originals = dict(cfg.rewards)
  snapshots = deepcopy(cfg.rewards)
  tracking = {
    k
    for k, t in originals.items()
    if t.func in (r.base.track_linear_velocity, r.base.track_angular_velocity)
  }
  r.configure_teacherfree_rewards(cfg)
  assert len(cfg.rewards) == len(originals) + 2
  for key in originals.keys() - tracking:
    assert cfg.rewards[key] is originals[key]
    assert cfg.rewards[key] == snapshots[key]
  assert {cfg.rewards[k].weight for k in tracking} == {2.0}
  assert cfg.rewards["teacherfree_linear_huber"].weight == -0.20
  assert cfg.rewards["teacherfree_yaw_huber"].weight == -0.10
  for key in tracking:
    assert cfg.rewards[key].params == {
      k: v for k, v in originals[key].params.items() if k != "std"
    }
  with pytest.raises(ValueError):
    r.configure_teacherfree_rewards(cfg)


def test_reward_manager_owns_single_dt_multiplication():
  cfg = black_teacher_omni_env_cfg()
  r.configure_teacherfree_rewards(cfg)
  functions = (
    r.track_linear_velocity,
    r.track_angular_velocity,
    r.linear_velocity_huber_cost,
    r.yaw_velocity_huber_cost,
  )
  terms = {k: v for k, v in cfg.rewards.items() if v.func in functions}
  env = fake_env([[0, 0, 0]])
  # Exercise the real compute path without scene resolution or simulation startup.
  manager = object.__new__(RewardManager)
  manager._env = env
  manager._scale_by_dt = True
  manager._term_names = list(terms)
  manager._term_cfgs = list(terms.values())
  manager._reward_buf = torch.zeros(1, dtype=torch.float64)
  manager._step_reward = torch.zeros(1, 4, dtype=torch.float64)
  manager._episode_sums = {key: torch.zeros(1, dtype=torch.float64) for key in terms}
  assert manager.compute(0.02).item() == pytest.approx(0.08)
  assert manager.compute(0.04).item() == pytest.approx(0.16)
  manager._scale_by_dt = False
  assert manager.compute(0.04).item() == pytest.approx(4.0)


def test_logging_masks_are_disjoint_and_exhaustive():
  masks = r.command_condition_masks(
    torch.tensor([[0.0, 0, 0], [0.55, 0, 0], [2.0, 0, 0]])
  )
  torch.testing.assert_close(sum(m.int() for m in masks.values()), torch.ones(3).int())
  assert masks["low"].tolist() == [True, False, False]
  assert masks["high"].tolist() == [False, False, True]
