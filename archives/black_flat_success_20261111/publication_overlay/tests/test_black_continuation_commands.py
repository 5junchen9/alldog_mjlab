from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.black_teacher.continuation_command import (
  STAGE_WEIGHTS,
  ContinuationVelocityCommandCfg,
  sample_targets,
)
from mjlab.tasks.velocity.config.black_teacher.continuation_env_cfg import (
  black_continuation_env_cfg,
)
from mjlab.tasks.velocity.config.black_teacher.env_cfgs import (
  black_teacher_omni_env_cfg,
)


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_distribution_and_bounds(stage):
  torch.manual_seed(123)
  target, bucket = sample_targets(stage, 40000, "cpu")
  actual = torch.bincount(bucket, minlength=7).float() / len(bucket)
  torch.testing.assert_close(
    actual, torch.tensor(STAGE_WEIGHTS[stage]), atol=0.012, rtol=0
  )
  low = target[bucket == 0]
  assert ((low != 0).sum(-1) == 1).all()
  assert low.abs().max() <= 0.30
  assert low.abs().sum(-1).min() >= 0.05
  high = target[bucket == 2].abs()
  assert ((high != 0).sum(-1) == 1).all()
  assert (high[:, :2][high[:, :2] > 0] >= 0.5).all()
  assert (high[:, 2][high[:, 2] > 0] >= 0.3).all()
  mixed = target[bucket == 1]
  assert mixed[:, :2].norm(dim=-1).min() >= 0.1 - 1e-6
  assert mixed[:, :2].norm(dim=-1).max() <= 0.5 + 1e-6
  assert mixed[:, 2].abs().max() <= 0.3
  assert (target[bucket == 6] == 0).all()
  assert (target[:, :2].abs() <= 1).all()
  assert (target[:, 2].abs() <= 0.5).all()
  if stage == "stage2":
    assert (target[bucket == 4].abs() == torch.tensor([1.0, 1.0, 0.5])).all()


def make_command(n=20000):
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
  return ContinuationVelocityCommandCfg(
    stage="stage2",
    entity_name="robot",
    resampling_time_range=(3, 8),
    ranges=ContinuationVelocityCommandCfg.Ranges((-1, 1), (-1, 1), (-0.5, 0.5)),
  ).build(cast(ManagerBasedRlEnv, env))


def test_transition_split_ramp_and_partial_update():
  torch.manual_seed(777)
  cmd = make_command()
  cmd.initialized[:] = True
  cmd.vel_command_b[:] = torch.tensor([0.8, 0.6, 0.4])
  cmd._resample_command(torch.arange(cmd.num_envs))
  transition = cmd.task_class == 5
  kinds = cmd.transition_kind[transition]
  actual = torch.bincount(kinds, minlength=3).float() / len(kinds)
  torch.testing.assert_close(
    actual, torch.tensor([0.5, 0.25, 0.25]), atol=0.035, rtol=0
  )
  assert abs(float(cmd.abrupt[transition].float().mean()) - 0.3) < 0.035
  reverse = cmd.transition_kind == 0
  torch.testing.assert_close(
    cmd.target_command[reverse],
    torch.tensor([-0.8, -0.6, -0.4]).expand(int(reverse.sum().item()), -1),
  )
  assert (cmd.target_command[cmd.transition_kind == 1] == 0).all()
  ramp_start = (cmd.transition_kind == 2) & ~cmd.abrupt
  assert (cmd.command[ramp_start] == 0).all()
  before = cmd.command.clone()
  cmd._update_command(torch.arange(10000))
  torch.testing.assert_close(cmd.command[10000:], before[10000:])
  assert (cmd.command - before)[:, :2].norm(dim=-1).max() <= 0.020001
  assert (cmd.command - before)[:, 2].abs().max() <= 0.010001


def test_reset_start_and_per_env_elapsed_time():
  cmd = make_command(128)
  cmd.reset(torch.arange(128))
  assert (cmd.transition_kind[cmd.task_class == 5] == 2).all()
  cmd.time_left[:] = 3
  dt = torch.full((128,), 0.02)
  dt[:10] = 0
  before = cmd.command.clone()
  cmd.compute(dt)
  torch.testing.assert_close(cmd.transition_age, dt)
  torch.testing.assert_close(cmd.command[:10], before[:10])
  cmd.reset(torch.arange(10))
  assert (cmd.transition_age[:10] == 0).all()
  torch.testing.assert_close(cmd.transition_age[10:], dt[10:])
  assert cmd.sample_counts.sum().item() == 138


def test_config_preserves_omni():

  original = black_teacher_omni_env_cfg()
  cfg = black_continuation_env_cfg("stage2")
  assert cfg.rewards == original.rewards
  assert cfg.events == original.events
  assert cfg.actions == original.actions
  assert cfg.observations == original.observations
  assert cfg.auto_reset
  assert cast(ContinuationVelocityCommandCfg, cfg.commands["twist"]).stage == "stage2"
  assert "history" not in cfg.observations
  with pytest.raises(ValueError):
    black_continuation_env_cfg("bad")
