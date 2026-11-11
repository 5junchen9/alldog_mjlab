"""CPU-only configuration and fake-environment tests; no physics or training."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.black_teacher.env_cfgs import (
  black_teacher_omni_env_cfg,
  black_teacher_speed_env_cfg,
)
from mjlab.tasks.velocity.config.black_teacher.omni_command import (
  OmniVelocityCommandCfg,
)
from mjlab.tasks.velocity.config.black_teacher.speed_command import (
  ConfiguredOmniVelocityCommandCfg,
)


def test_speed_projects_require_explicit_configuration_and_are_isolated():
  omni = black_teacher_omni_env_cfg()
  for suffix in ("LowSpeed", "HighSpeed"):
    task = f"Mjlab-Velocity-Flat-Black-Teacher-{suffix}"
    for play in (False, True):
      cfg = load_env_cfg(task, play=play)
      command = cfg.commands["twist"]
      assert isinstance(command, ConfiguredOmniVelocityCommandCfg)
      assert command.min_speeds is None
      with pytest.raises(ValueError, match="explicit min_speeds"):
        command.build(cast(Any, None))
    assert load_rl_cfg(task).experiment_name != "black_teacher_omni_flat"
  low, high = black_teacher_speed_env_cfg("low"), black_teacher_speed_env_cfg("high")
  assert {name: asdict(term) for name, term in low.rewards.items()} == {
    name: asdict(term) for name, term in omni.rewards.items()
  }
  assert low.rewards is not high.rewards
  assert low.commands["twist"] is not high.commands["twist"]
  assert type(omni.commands["twist"]) is OmniVelocityCommandCfg
  assert load_rl_cfg("Mjlab-Velocity-Flat-Black-Teacher-LowSpeed").experiment_name != (
    load_rl_cfg("Mjlab-Velocity-Flat-Black-Teacher-HighSpeed").experiment_name
  )


@pytest.mark.parametrize(
  "minimum,bounds",
  [
    ((0.0, 0.05, 0.05), (-0.1, 0.1)),
    ((0.2, 0.05, 0.05), (-0.1, 0.1)),
    ((0.05, 0.05, 0.05), (-0.2, 0.1)),
    ((float("nan"), 0.05, 0.05), (-0.1, 0.1)),
    ((0.05, 0.05, 0.05), (-float("inf"), float("inf"))),
  ],
)
def test_speed_bounds_rejected_before_environment_use(minimum, bounds):
  cfg = black_teacher_speed_env_cfg("low").commands["twist"]
  assert isinstance(cfg, ConfiguredOmniVelocityCommandCfg)
  cfg.min_speeds = minimum
  cfg.ranges.lin_vel_x = bounds
  with pytest.raises(ValueError):
    cfg.build(cast(Any, None))


def test_explicit_sub_point_one_speeds_keep_all_categories_and_partial_resets():
  torch.manual_seed(137)
  count = 8192
  cfg = black_teacher_speed_env_cfg("low").commands["twist"]
  assert isinstance(cfg, ConfiguredOmniVelocityCommandCfg)
  cfg.min_speeds = (0.05, 0.05, 0.05)
  cfg.ranges.lin_vel_x = cfg.ranges.lin_vel_y = cfg.ranges.ang_vel_z = (-0.08, 0.08)
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
  command = cfg.build(cast(Any, env))
  command._resample_command(torch.arange(count))
  values = command.command.clone()
  pure = (values != 0).sum(dim=1) == 1
  assert pure.any()
  magnitudes = values[pure].abs().amax(dim=1)
  assert torch.all((magnitudes >= 0.05) & (magnitudes <= 0.08))
  assert magnitudes.min() < 0.055 and magnitudes.max() > 0.075
  assert values.abs().max() <= 0.08
  command._update_metrics()
  coverage = env.extras["log"]
  assert len(coverage) == 8
  expected = (0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.15)
  for value, probability in zip(coverage.values(), expected, strict=True):
    assert abs(value.item() - probability) < 0.025
  command._resolve_command(torch.zeros(count))
  torch.testing.assert_close(command.command, values)
  command._resample_command(torch.tensor([0, 2, 4]))
  torch.testing.assert_close(command.command[5:], values[5:])


def test_equal_positive_speed_bounds_are_valid():
  cfg = black_teacher_speed_env_cfg("high").commands["twist"]
  assert isinstance(cfg, ConfiguredOmniVelocityCommandCfg)
  cfg.min_speeds = (1.0, 1.0, 0.5)
  cfg.validate_speeds()
