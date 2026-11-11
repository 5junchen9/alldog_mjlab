"""检查同一教师的实际指令覆盖与局部重采样。"""

from types import SimpleNamespace
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.black_teacher.env_cfgs import (
  black_teacher_omni_env_cfg,
)
from mjlab.tasks.velocity.config.black_teacher.omni_command import OmniVelocityCommand


def test_omni_command_covers_all_directions_without_heading_override():
  torch.manual_seed(42)
  count = 4096
  cfg = black_teacher_omni_env_cfg()
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_lin_vel_b=torch.zeros(count, 3),
      root_link_ang_vel_b=torch.zeros(count, 3),
      heading_w=torch.zeros(count),
    )
  )
  env = SimpleNamespace(
    num_envs=count,
    device="cpu",
    step_dt=0.02,
    scene={"robot": robot},
    extras={"log": {}},
  )
  command = cast(
    OmniVelocityCommand, cfg.commands["twist"].build(cast(ManagerBasedRlEnv, env))
  )
  command._resample_command(torch.arange(count))
  before = command.command.clone()
  command._resolve_command(torch.randn(count))
  torch.testing.assert_close(command.command, before)
  command._update_metrics()
  coverage = env.extras["log"]
  assert len(coverage) == 8
  assert all(value > 0.02 for value in coverage.values())
  torch.testing.assert_close(sum(coverage.values()), torch.tensor(1.0))
  assert command.command[:, :2].abs().max() <= 1.0
  assert command.command[:, 2].abs().max() <= 0.5
  command._resample_command(torch.tensor([0, 2, 4]))
  torch.testing.assert_close(command.command[5:], before[5:])
  assert "command_vel" not in cfg.curriculum
