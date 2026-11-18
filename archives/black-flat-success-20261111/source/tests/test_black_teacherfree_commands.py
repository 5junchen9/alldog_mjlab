from dataclasses import fields
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.black_teacher import teacherfree_command as module
from mjlab.tasks.velocity.config.black_teacher.env_cfgs import (
  black_teacher_omni_env_cfg,
)
from mjlab.tasks.velocity.config.black_teacher.teacherfree_command import (
  OMNI_WEIGHTS,
  RECOVERY_BUCKET,
  TeacherFreeVelocityCommandCfg,
  allocate_profiles,
  sample_targets,
)
from mjlab.tasks.velocity.config.black_teacher.teacherfree_env_cfg import (
  black_teacherfree_env_cfg,
)


def make_command(n=64, abrupt_probability=0.5):
  data = SimpleNamespace(
    heading_w=torch.zeros(n),
    root_link_pos_w=torch.zeros(n, 3),
    root_link_lin_vel_b=torch.zeros(n, 3),
    root_link_ang_vel_b=torch.zeros(n, 3),
  )
  env = SimpleNamespace(
    num_envs=n,
    device="cpu",
    step_dt=0.02,
    scene={"robot": SimpleNamespace(data=data)},
    extras={"log": {}},
    episode_length_buf=torch.ones(n),
  )
  return TeacherFreeVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3, 8),
    abrupt_probability=abrupt_probability,
    ranges=TeacherFreeVelocityCommandCfg.Ranges((-1, 1), (-1, 1), (-0.5, 0.5)),
  ).build(cast(ManagerBasedRlEnv, env))


def force_recovery(monkeypatch):
  def sample(profiles, ranges=None):
    del ranges
    return (
      torch.tensor([1.0, -1.0, 0.5]).expand(len(profiles), -1).clone(),
      torch.full_like(profiles, RECOVERY_BUCKET),
    )

  monkeypatch.setattr(module, "sample_targets", sample)


def test_fixed_profile_counts_and_reset_persistence():
  profiles = allocate_profiles(4096, "cpu", seed=31)
  assert torch.bincount(profiles).tolist() == [1434, 1433, 1229]
  assert torch.equal(profiles, allocate_profiles(4096, "cpu", seed=31))
  assert not torch.equal(profiles, allocate_profiles(4096, "cpu", seed=32))
  for n in range(1, 15):
    assert len(allocate_profiles(n, "cpu")) == n
  cmd = make_command(4096)
  before = cmd.profile_id.clone()
  cmd.reset(torch.arange(4096))
  cmd.compute(17.0)
  cmd.reset(torch.tensor([19, 0, 118]))
  assert torch.equal(before, cmd.profile_id)
  expected = torch.bincount(before).float() * 17
  torch.testing.assert_close(cmd.profile_held_seconds, expected, rtol=1e-4, atol=0.1)


@pytest.mark.parametrize("profile", [0, 2])
def test_low_high_distribution_ranges_and_six_axes(profile):
  torch.manual_seed(123)
  target, bucket = sample_targets(torch.full((60000,), profile))
  expected = torch.tensor(
    [0.7, 0.2, 0, 0, 0, 0, 0.1] if profile == 0 else [0, 0, 0, 0, 0.5, 0.5, 0]
  )
  actual = torch.bincount(bucket, minlength=7).float() / len(bucket)
  torch.testing.assert_close(actual, expected, atol=0.01, rtol=0)
  pure = target[bucket == (0 if profile == 0 else 4)]
  assert ((pure != 0).sum(-1) == 1).all()
  direction = pure.abs().argmax(-1) * 2 + (pure.sum(-1) < 0)
  shares = torch.bincount(direction, minlength=6).float() / len(pure)
  torch.testing.assert_close(shares, torch.full((6,), 1 / 6), atol=0.01, rtol=0)
  if profile == 0:
    assert pure.abs().max() <= 0.3
    assert pure.abs().sum(-1).min() >= 0.05
    mixed = target[bucket == 1]
    assert mixed[:, :2].norm(dim=-1).min() >= 0.1 - 1e-6
    assert mixed[:, :2].norm(dim=-1).max() <= 0.5 + 1e-6
    assert mixed[:, 2].abs().max() <= 0.3
    assert (target[bucket == 6] == 0).all()
  else:
    xy = pure[:, :2].abs()
    yaw = pure[:, 2].abs()
    assert (xy[xy > 0] >= 0.5).all() and xy.max() <= 1
    assert (yaw[yaw > 0] >= 0.3).all() and yaw.max() <= 0.5
    assert (target[bucket == RECOVERY_BUCKET].abs() == torch.tensor([1, 1, 0.5])).all()


def test_omni_exact_category_distribution_and_original_minimum():
  torch.manual_seed(321)
  target, bucket = sample_targets(torch.ones(80000, dtype=torch.long))
  pure = bucket == 2
  axis = target.abs().argmax(-1)
  positive = target.sum(-1) > 0
  category = torch.where(bucket == 3, 6, 7)
  for index, dim, sign in (
    (0, 0, True),
    (1, 0, False),
    (2, 2, True),
    (3, 2, False),
    (4, 1, True),
    (5, 1, False),
  ):
    category[pure & (axis == dim) & (positive == sign)] = index
  actual = torch.bincount(category, minlength=8).float() / len(category)
  torch.testing.assert_close(actual, torch.tensor(OMNI_WEIGHTS), atol=0.01, rtol=0)
  assert target[pure].abs().sum(-1).min() >= 0.1
  # Original Omni clamps samples near zero, creating real mass at exactly 0.1.
  assert (target[pure].abs().sum(-1) == 0.1).float().mean() > 0.09
  assert target[:, :2].abs().max() <= 1 and target[:, 2].abs().max() <= 0.5
  mixed = target[bucket == 3]
  assert mixed[:, :2].abs().min() < 0.01
  assert mixed[:, :2].abs().max() > 0.95


def test_cycle_targets_phase_timers_and_event_counts(monkeypatch):
  force_recovery(monkeypatch)
  cmd = make_command(4, abrupt_probability=1)
  cmd.reset(torch.arange(4))
  assert (cmd.cycle_phase == 0).all() and (cmd.time_left == 1).all()
  assert (cmd.command == 0).all()
  original = cmd.cycle_target.clone()
  cmd.compute(1.0)
  assert (cmd.cycle_phase == 1).all() and (cmd.time_left == 5).all()
  torch.testing.assert_close(cmd.command, original)
  cmd.compute(5.0)
  assert (cmd.cycle_phase == 2).all() and (cmd.time_left == 5).all()
  torch.testing.assert_close(cmd.command, -original)
  cmd.compute(5.0)
  assert (cmd.cycle_phase == 3).all() and (cmd.time_left == 3).all()
  assert (cmd.command == 0).all()
  assert cmd.sample_counts.sum() == 4
  cmd.compute(3.0)
  assert (cmd.cycle_phase == 1).all()  # No repeated initialization settle.
  assert cmd.sample_counts.sum() == 8
  torch.testing.assert_close(cmd.held_seconds[RECOVERY_BUCKET], torch.tensor(56.0))
  torch.testing.assert_close(cmd.phase_seconds, torch.tensor([4, 20, 20, 12.0]))


def test_ramp_actual_dt_zero_time_subsets_and_boundary_crossing(monkeypatch):
  force_recovery(monkeypatch)
  cmd = make_command(4, abrupt_probability=0)
  cmd.reset(torch.arange(4))
  cmd.compute(1.0)
  assert (cmd.command == 0).all()  # New target has zero elapsed ramp time.
  before = cmd.command.clone()
  cmd.compute(torch.tensor([0.0, 0.1, 0.2, 0.05]))
  delta = cmd.command - before
  torch.testing.assert_close(
    delta[:, :2].norm(dim=-1), torch.tensor([0, 0.1, 0.2, 0.05])
  )
  torch.testing.assert_close(delta[:, 2], torch.tensor([0, 0.05, 0.1, 0.025]))
  before = cmd.command.clone()
  times = cmd.time_left.clone()
  cmd.compute(0.3, torch.tensor([3, 1]))
  torch.testing.assert_close(cmd.command[[0, 2]], before[[0, 2]])
  torch.testing.assert_close(cmd.time_left[[0, 2]], times[[0, 2]])
  # Environment 0 spends 5s reaching boundary then only .25s reversing.
  cmd.compute(5.25, torch.tensor([0]))
  expected = torch.tensor([1.0, -1.0, 0.5])
  expected[:2] *= 1 - 0.25 / (2**0.5)
  expected[2] -= 0.125
  torch.testing.assert_close(cmd.command[0], expected)
  assert cmd.cycle_phase[0] == 2 and cmd.time_left[0] == 4.75


def test_reset_clears_only_selected_cycle_state(monkeypatch):
  force_recovery(monkeypatch)
  cmd = make_command(5, abrupt_probability=0)
  cmd.reset(torch.arange(5))
  cmd.compute(7.0)
  before = cmd.command.clone()
  phases = cmd.cycle_phase.clone()
  timers = cmd.time_left.clone()
  cmd.reset(torch.tensor([4, 1]))
  assert (cmd.cycle_phase[[4, 1]] == 0).all()
  assert (cmd.phase_age[[4, 1]] == 0).all()
  assert (cmd.command[[4, 1]] == 0).all()
  assert (cmd.target_command[[4, 1]] == 0).all()
  torch.testing.assert_close(cmd.command[[0, 2, 3]], before[[0, 2, 3]])
  assert torch.equal(cmd.cycle_phase[[0, 2, 3]], phases[[0, 2, 3]])
  torch.testing.assert_close(cmd.time_left[[0, 2, 3]], timers[[0, 2, 3]])


def test_cycle_mode_split_persistent_and_held_metrics(monkeypatch):
  force_recovery(monkeypatch)
  torch.manual_seed(77)
  cmd = make_command(10000)
  cmd.reset(torch.arange(cmd.num_envs))
  before = cmd.abrupt.clone()
  assert abs(cmd.abrupt.float().mean().item() - 0.5) < 0.02
  cmd.compute(6.25)
  assert torch.equal(cmd.abrupt, before)
  assert cmd.transition_mode_counts.sum() == cmd.num_envs
  log = cmd._env.extras["log"]
  assert log["TeacherFreeHeldShare/boundary_recovery"] == 1
  assert log["TeacherFreeEvents/boundary_recovery"] == cmd.num_envs
  assert log["TeacherFreePhase/reverse"] == 1


def test_config_preserves_original_omni_without_history_or_teacher():
  original, cfg = black_teacher_omni_env_cfg(), black_teacherfree_env_cfg()
  assert cfg.rewards == original.rewards
  assert cfg.events == original.events
  assert cfg.actions == original.actions
  assert cfg.observations == original.observations
  assert cfg.curriculum == original.curriculum
  for field in fields(cfg.scene):
    if field.name != "terrain":
      assert getattr(cfg.scene, field.name) == getattr(original.scene, field.name)
  assert cfg.scene.terrain is not None
  assert original.scene.terrain is not None
  for field in fields(cfg.scene.terrain):
    if field.name != "spec_fn":  # Each factory creates an equivalent new lambda.
      assert getattr(cfg.scene.terrain, field.name) == getattr(
        original.scene.terrain, field.name
      )
  assert cfg.auto_reset
  assert "history" not in cfg.observations
  assert cfg.commands["twist"].resampling_time_range == (3, 8)
  assert isinstance(cfg.commands["twist"], TeacherFreeVelocityCommandCfg)
  assert isinstance(original.commands["twist"], module.OmniVelocityCommandCfg)
  assert cfg.commands["twist"].ranges == original.commands["twist"].ranges
