from types import SimpleNamespace
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.black_teacher.candidate_command import (
  CandidateVelocityCommandCfg,
  low_magnitudes,
  ramp_command,
  sample_targets,
)


def test_low_bands_and_distribution():
  torch.manual_seed(351)
  v = low_magnitudes(100000, "cpu")
  assert v.min() >= 0.05 and v.max() <= 0.5
  assert abs(float((v < 0.15).float().mean()) - 0.4) < 0.01
  assert abs(float(((v >= 0.15) & (v < 0.3)).float().mean()) - 0.4) < 0.01
  commands, category = sample_targets("low", 30000, "cpu")
  assert commands[:, :2].norm(dim=-1).max() <= 0.5 + 1e-6
  assert commands[category == 6, 2].abs().max() <= 0.3
  expected = torch.tensor([0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10])
  actual = torch.bincount(category, minlength=8).float() / len(category)
  torch.testing.assert_close(actual, expected, atol=0.012, rtol=0)
  assert (commands[category == 7] == 0).all()


def test_high_boundary_and_ramp():
  torch.manual_seed(123)
  command, category = sample_targets("high", 20000, "cpu")
  mixed = command[category == 6].abs()
  assert (mixed[:, :2] >= 0.5).all() and (mixed[:, :2] <= 1).all()
  boundary = (mixed == torch.tensor([1.0, 1.0, 0.5])).all(dim=-1)
  assert 0.47 < float(boundary.float().mean()) < 0.53
  before = torch.tensor([[1.0, 1.0, 0.5]])
  after = ramp_command(before, -before, 0.02)
  assert (after - before)[:, :2].norm() <= 0.020001
  torch.testing.assert_close(after[:, 2], torch.tensor([0.49]))


def test_high_partial_resample_and_reversal():
  n = 64
  data = SimpleNamespace(heading_w=torch.zeros(n))
  env = SimpleNamespace(
    num_envs=n,
    device="cpu",
    step_dt=0.02,
    scene={"robot": SimpleNamespace(data=data)},
    extras={"log": {}},
  )
  cfg = CandidateVelocityCommandCfg(
    profile="high",
    entity_name="robot",
    resampling_time_range=(3, 8),
    ranges=CandidateVelocityCommandCfg.Ranges((-1, 1), (-1, 1), (-0.5, 0.5)),
    reversal_probability=1.0,
    abrupt_probability=0.0,
  )
  cmd = cfg.build(cast(ManagerBasedRlEnv, env))
  cmd._resample_command(torch.arange(n))
  before = cmd.command.clone()
  previous = cmd.target_command.clone()
  cmd._resample_command(torch.arange(32))
  torch.testing.assert_close(cmd.command, before)
  assert cmd.reversal[:32].any()
  reversed_ids = cmd.reversal[:32].nonzero().flatten()
  torch.testing.assert_close(cmd.target_command[reversed_ids], -previous[reversed_ids])
  cmd._update_command(torch.arange(32))
  torch.testing.assert_close(cmd.command[32:], before[32:])
  assert (cmd.command[:32] - before[:32])[:, :2].norm(dim=-1).max() <= 0.020001
