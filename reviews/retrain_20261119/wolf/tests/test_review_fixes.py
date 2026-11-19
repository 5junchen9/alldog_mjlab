"""轮足专项审查反例，使用原始几何平面和独立评分。"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from wolf.curriculum import SPECIALTIES, CourseCommand, configure
from wolf.eval_protocol import SEEDS, panel
from wolf.eval_score import geometry_contract, in_surface, score_trial
from wolf.rewards import (
  ROUTE_COMPLETION_BONUS,
  RouteCompletionBonus,
  track_linear_velocity_x,
  verify_completion_bound,
)
from wolf.tests.test_contract import flat_rows


@pytest.mark.parametrize("course", ["broken_bridge", "plum_stakes"])
@pytest.mark.parametrize("direction", [-1, 1])
def test_mesh_ramp_support_passes_but_core_floor_fails(course, direction):
  spec = panel(course, SEEDS[0], 0)[0]
  spec = {**spec, "direction": direction, "duration": 52}
  geo = geometry_contract(spec)
  ramps = [s for s in geo["surfaces"] if abs(s[4]) > 1e-4]
  assert len(ramps) >= 2
  surface = ramps[0 if direction > 0 else -1]
  x, y = (surface[0] + surface[1]) / 2, (surface[2] + surface[3]) / 2
  z = -(surface[4] * x + surface[5] * y + surface[7]) / surface[6]
  p = np.array([[x, y, z]] * 4)
  assert in_surface(p, surface).all()
  rows = flat_rows(spec)
  for row in rows:
    row["command"] = row["history_command"] = row["velocity"] = [0.0, 0.0, 0.0]
    row["angular"] = [0.0, 0.0, 0.0]
    exit_x = geo["length"] - 0.5 if direction > 0 else 0.5
    row["root"] = [exit_x, 0.0, 0.4]
    row["contact_points"] = [[exit_x, 0.0, 0.0]] * 4
  rows[0]["root"] = [x, 0.0, 0.4]
  rows[0]["contact_points"] = p.tolist()
  assert score_trial(rows, spec, geo)["status"] == "pass"
  bad = deepcopy(rows)
  core_x = sum(geo["core"]) / 2
  bad[1]["root"] = [core_x, 0.0, 0.4]
  bad[1]["contact_points"] = [[core_x, 0.0, 0.0]] * 4
  assert score_trial(bad, spec, geo)["status"] == "fail"


def test_all_six_rewards_exceed_verified_discounted_positive_tail_and_pay_once():
  for course in SPECIALTIES:
    cfg = configure(course, 32, 19)
    assert verify_completion_bound(cfg) == pytest.approx(5.5)
    assert ROUTE_COMPLETION_BONUS > 5.5
    assert {
      "route_completion_bonus",
      "route_progress_shaping",
      "route_stall_penalty",
    } <= cfg.rewards.keys()
  cmd = SimpleNamespace(
    flat_mask=torch.tensor([False, True]),
    passed=torch.tensor([True, True]),
    failed=torch.tensor([False, False]),
    evidence=lambda: None,
  )
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    step_dt=0.02,
    command_manager=SimpleNamespace(get_term=lambda _: cmd),
  )
  reward = RouteCompletionBonus(None, env)
  assert reward(env).tolist() == [ROUTE_COMPLETION_BONUS / 0.02, 0.0]
  assert reward(env).eq(0).all()
  reward.reset(torch.tensor([0]))
  cmd.failed[0] = True
  assert reward(env).eq(0).all()


def test_phase_speed_and_composite_commands_and_low_speed_standing_cost():
  command = object.__new__(CourseCommand)
  command._env = SimpleNamespace(device="cpu", num_envs=4096)
  command.target = torch.zeros(4096, 3)
  command.flat_mask = torch.zeros(4096, dtype=torch.bool)
  command.specialty_speed = torch.zeros(4096)
  ids = torch.arange(4096)
  for course in ("flat_foundation", "flat_axes", "general", "stairs"):
    command.cfg = SimpleNamespace(course=course)
    command._resample_command(ids)
    if course == "flat_foundation":
      assert command.target[:, 0].abs().max() <= 0.3
    else:
      assert (command.target.abs().gt(0.01).sum(-1) > 1).any()
      assert command.target[:, 0].abs().max() > 0.7
  env = SimpleNamespace(
    command_manager=SimpleNamespace(
      get_command=lambda _: torch.tensor([[0.2, 0.0, 0.0]])
    ),
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(root_link_lin_vel_b=torch.zeros(1, 3))
      )
    },
  )
  assert track_linear_velocity_x(env, "twist").item() < 0.5


def test_safety_and_general_start_at_first_rough_level():
  cfg = configure("general", 32, 19)
  assert "bad_orientation" in cfg.terminations
  assert "out_of_terrain_bounds" in cfg.terminations
  assert cfg.scene.terrain.max_init_terrain_level == 1
  cfg = configure("high_wall", 32, 19, max_level=1)
  assert "bad_orientation" in cfg.terminations
  assert "route_failed" in cfg.terminations


def test_qualification_requires_every_mixed_command_component():
  spec = next(
    row for row in panel("flat_axes", SEEDS[0]) if row["target"][0] and row["target"][2]
  )
  rows = flat_rows(spec)
  assert score_trial(rows, spec, None)["status"] == "pass"
  for row in rows:
    row["angular"][2] = 0.0  # 只前进、不转向不能通过复合命令验收。
  assert score_trial(rows, spec, None)["status"] == "fail"
