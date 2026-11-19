"""课程范围、命令阶段和超时不能晋级的CPU反例。"""

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from curriculum import (
  RESERVED_SEEDS,
  configure,
  flat_commands,
  strict_passage_book,
  training_seed,
)
from runtime import native_imports

native_imports()
from parameters import PARAMETERS  # noqa: E402
from passage_course import POLICY, PassageCourse  # noqa: E402
from specialty_env import make_specialty_configs  # noqa: E402


class Scene(dict):
  terrain: Any


def test_private_parameters_and_default_foundation_do_not_change_old_configs():
  cfg, _ = make_specialty_configs(
    "stairs", num_envs=32, seed=73, course_parameters=PARAMETERS, passage_policy=POLICY
  )
  old = copy.deepcopy(cfg.commands["twist"].course_parameters)
  receipt = configure(cfg)
  assert receipt["flat_fraction"] == 1.0
  assert cfg.commands["twist"].course_parameters["settle_steps"] == 50
  assert cfg.commands["twist"].course_parameters["finish_max_angular_speed"] == 0.12
  assert PARAMETERS == old and PARAMETERS["settle_steps"] == 6


@pytest.mark.parametrize(
  "course,flat_count", [("flat_foundation", 32), ("flat_axes", 32), ("stairs", 8)]
)
def test_built_command_uses_all_flat_first_and_keeps_flat_replay(
  monkeypatch, course, flat_count
):
  import specialty_env

  class CommandStub:
    def __init__(self, cfg, env):
      self._env, self.device, self.num_envs = env, "cpu", 32
      self.flat_mask = torch.arange(32) < 8
      self.vel_command_b = torch.zeros(32, 3)

    def compute(self, dt, env_ids=None):
      self.vel_command_b.fill_(9)

  monkeypatch.setattr(specialty_env, "SpecialtyCommand", CommandStub)
  cfg, _ = make_specialty_configs(
    "stairs", num_envs=32, seed=73, course_parameters=PARAMETERS, passage_policy=POLICY
  )
  configure(cfg, course)
  terrain = SimpleNamespace(
    terrain_types=torch.ones(32, dtype=torch.long),
    terrain_levels=torch.zeros(32, dtype=torch.long),
    terrain_origins=torch.randn(10, 7, 3),
    env_origins=torch.zeros(32, 3),
  )
  env = SimpleNamespace(
    scene=SimpleNamespace(terrain=terrain),
    episode_length_buf=torch.full((32,), 100, dtype=torch.long),
  )
  term = cfg.commands["twist"].build(env)
  term.compute(0.02)
  assert int(term.flat_mask.sum()) == flat_count
  assert term.vel_command_b[term.flat_mask].abs().max() <= 0.5
  if course == "stairs":
    assert term.vel_command_b[~term.flat_mask].eq(9).all()
  else:
    assert terrain.terrain_types.eq(0).all()
    torch.testing.assert_close(
      terrain.env_origins, terrain.terrain_origins[0, 0].expand(32, -1)
    )


def test_axis_chain_starts_reverses_and_parks_without_mixed_commands():
  age = torch.tensor([0, 99, 100, 349, 350, 599, 600, 899, 900])
  actual = flat_commands(age, torch.zeros(len(age), dtype=torch.long), "flat_axes")
  assert actual[:, 0].tolist() == pytest.approx([0, 0, -0.1, -0.1, 0.1, 0.1, 0, 0, 0])
  assert actual[:, 1:].eq(0).all()
  cells = flat_commands(torch.full((12,), 100), torch.arange(12), "stairs")
  assert len(set(map(tuple, cells.tolist()))) == 12
  assert cells.ne(0).sum(-1).eq(1).all()


@pytest.mark.parametrize("course", ["flat_foundation", "flat_axes", "stairs"])
def test_real_4096_command_resampling_skips_old_1024_lane_augmentation(course):
  """直接调用真实命令继承链；不构造仿真或替换原生重采样实现。"""
  cfg, _ = make_specialty_configs(
    "stairs",
    num_envs=4096,
    seed=73,
    course_parameters=PARAMETERS,
    passage_policy=POLICY,
  )
  configure(cfg, course)
  scene = Scene(robot=SimpleNamespace())
  scene.terrain = SimpleNamespace(
    terrain_types=torch.zeros(4096, dtype=torch.long),
    terrain_levels=torch.zeros(4096, dtype=torch.long),
    terrain_origins=torch.randn(11, 7, 3),
    env_origins=torch.zeros(4096, 3),
  )

  class OldCoverage:
    def resample(self, *args):
      raise AssertionError("v2必须跳过历史1024条覆盖增样器")

  env = SimpleNamespace(
    num_envs=4096,
    device="cpu",
    scene=scene,
    common_step_counter=100,
    episode_length_buf=torch.full((4096,), 100, dtype=torch.long),
    specialty446_state=SimpleNamespace(flat_coverage=OldCoverage()),
  )
  term = cfg.commands["twist"].build(env)
  term._resample_command(torch.arange(4096))
  # 再测高编号子集；不能依赖首次初始化恰好没有抽中旧纯轴事件。
  term._resample_command(torch.arange(1024, 4096))
  term.publish()
  assert term.initialized.all()
  assert term.time_left.gt(0).all()
  flat = term.flat_mask
  torch.testing.assert_close(
    term.command[flat],
    flat_commands(env.episode_length_buf[flat], term.training_choice[flat], course),
  )
  assert int(flat.sum()) == (1024 if course == "stairs" else 4096)
  if course == "stairs":
    assert term.target_command[~flat, 0].ge(0.15).all()
    assert term.target_command[~flat, 0].le(0.35).all()
    assert term.target_command[~flat, 1:].eq(0).all()


def test_foundation_includes_quiet_start_and_stop():
  actual = flat_commands(
    torch.tensor([0, 100, 399, 400]),
    torch.zeros(4, dtype=torch.long),
    "flat_foundation",
  )
  assert actual[:, 0].tolist() == pytest.approx([0, 0.3, 0.3, 0])


def test_timeout_pass_cannot_advance_private_stair_course():
  policy: dict = copy.deepcopy(POLICY)
  policy.update(minimum_trials=2, window=2)
  state = SimpleNamespace(
    passage_course=PassageCourse(policy),
    command=SimpleNamespace(cfg=SimpleNamespace(passage_course_policy=policy)),
  )
  strict_passage_book(state)
  book = state.passage_course
  for stamp in range(1, 5):
    level, angle, direction = book.assign(stamp, stamp)
    # 固定headon，证明timeout再多也不能推动高度。
    book.active[stamp] = (stamp, level, 0, direction)
    book.vote(
      stamp,
      stamp,
      level,
      0,
      direction,
      "pass_timeout_no_stop",
      dict(
        ordered_treads=12,
        landing_contact=True,
        first_verified_pass_tick=stamp + 1,
        closed_tick=stamp + 2,
      ),
    )
    assert not book.advance_once(stamp + 2)
  assert book.current == 0
  assert book.outcomes["timeout_no_pass"] == 4
  assert not book.outcomes["pass_timeout_no_stop"]


def test_training_and_evaluation_seeds_are_disjoint():
  training_seed(73)
  for seed in RESERVED_SEEDS:
    with pytest.raises(ValueError, match="训练seed"):
      training_seed(seed)
