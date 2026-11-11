"""CPU tests for candidate kernels, smoothing, and original state preservation."""

from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.black_teacher import candidate_rewards as r


def fake_env(commands) -> ManagerBasedRlEnv:
  command = torch.tensor(commands, dtype=torch.float32)
  n = len(commands)
  data = NS(
    root_link_lin_vel_b=torch.zeros(n, 3),
    root_link_ang_vel_b=torch.zeros(n, 3),
    joint_pos=torch.ones(n, 2) * 0.1,
  )
  return cast(
    ManagerBasedRlEnv,
    NS(
      command_manager=NS(get_command=lambda _: command),
      scene={"robot": NS(data=data)},
      extras={"log": {}},
      step_dt=0.02,
    ),
  )


def test_tracking_exact_monotonic_and_original_off_axis():
  env = fake_env([[0, 0, 0], [0.1, 0, 0.05], [0.2, 0, 0.1]])
  for fn in (r.track_linear_velocity_fine, r.track_angular_velocity_fine):
    value = fn(env, "velocity")
    assert value[0] == 1
    assert torch.all(value[:-1] > value[1:])
    assert torch.isfinite(value).all()
  env = fake_env([[0, 0, 0]])
  env.scene["robot"].data.root_link_lin_vel_b[:, 2] = 0.4
  env.scene["robot"].data.root_link_ang_vel_b[:, :2] = 0.2
  torch.testing.assert_close(
    r.track_linear_velocity_fine(env, "velocity"),
    r.base.track_linear_velocity(env, 0.5, "velocity"),
  )
  torch.testing.assert_close(
    r.track_angular_velocity_fine(env, "velocity"),
    r.base.track_angular_velocity(env, 0.5**0.5, "velocity"),
  )


def test_huber_planar_and_yaw_semantics():
  env = fake_env([[0, 0, 0], [0.5, -0.5, 0.25], [1, -1, 0.5]])
  torch.testing.assert_close(
    r.linear_velocity_huber_cost(env, "v"), torch.tensor([0, 0.25, 0.75])
  )
  torch.testing.assert_close(
    r.yaw_velocity_huber_cost(env, "v"), torch.tensor([0, 0.03125, 0.09375])
  )
  env.scene["robot"].data.root_link_lin_vel_b[:, 2] = 100
  assert r.linear_velocity_huber_cost(env, "v")[0] == 0
  with pytest.raises(ValueError):
    r.huber(torch.zeros(1), 0)


def test_gate_endpoints_continuity_and_gradients():
  x = torch.tensor([-0.01, 0, 0.5, 1, 1.01], requires_grad=True)
  value = r.smoothstep(x)
  torch.testing.assert_close(value, torch.tensor([0, 0, 0.5, 1, 1]))
  value.sum().backward()
  torch.testing.assert_close(x.grad, torch.tensor([0, 0, 1.5, 0, 0]))
  env = fake_env([[0, 0, 0], [0, 0, 0.025], [0.03, 0.04, 0]])
  torch.testing.assert_close(r.command_gate(env, "v"), torch.tensor([0, 0.5, 1]))
  edge = r.command_gate(fake_env([[0.05 - 1e-6, 0, 0], [0.05, 0, 0]]), "v")
  assert abs(edge[0] - edge[1]) < 1e-6


@pytest.mark.parametrize("name", ["feet_clearance", "feet_slip", "soft_landing"])
def test_stateless_wrappers_call_original_once(monkeypatch, name):
  env = fake_env([[0, 0, 0], [0.025, 0, 0], [0.05, 0, 0]])
  original = Mock(return_value=torch.ones(3) * 2)
  monkeypatch.setattr(r.base, name, original)
  args = {"env": env, "command_name": "v"}
  if name == "feet_clearance":
    args.update(target_height=0.1, height_sensor_name="height")
  else:
    args["sensor_name"] = "feet"
  torch.testing.assert_close(getattr(r, name)(**args), torch.tensor([0.0, 1.0, 2.0]))
  original.assert_called_once()
  assert -1.0 in original.call_args.args


def test_swing_original_state_machine_runs_once_and_resets():
  env = fake_env([[0.025, 0, 0]])
  contact = NS(data=NS(found=torch.zeros(1, 2)))
  contact.compute_first_contact = Mock(return_value=torch.zeros(1, 2).bool())
  cast(dict[str, object], env.scene)["feet"] = contact
  cast(dict[str, object], env.scene)["height"] = NS(
    data=NS(heights=torch.tensor([[0.1, 0.15]]))
  )
  term = object.__new__(r.feet_swing_height)
  term.peak_heights = torch.zeros(1, 2)
  term.step_dt = env.step_dt
  args = (env, "feet", "height", 0.1, "v", 0.05)
  assert term(*args).item() == 0
  torch.testing.assert_close(term.peak_heights, torch.tensor([[0.1, 0.15]]))
  contact.data.found[:] = 1
  contact.compute_first_contact.return_value = torch.ones(1, 2).bool()
  torch.testing.assert_close(term(*args), torch.tensor([0.125]))
  assert contact.compute_first_contact.call_count == 2
  assert term.peak_heights.count_nonzero() == 0
  term.peak_heights[:] = 1
  term.reset()
  assert term.peak_heights.count_nonzero() == 0


def test_posture_anchors_and_smooth_boundary():
  speeds = [0, 0.5 - 1e-6, 0.5, 0.5 + 1e-6, 1.5, 2]
  env = fake_env([[s, 0, 0] for s in speeds])
  term = object.__new__(r.variable_posture)
  term.std_standing = torch.tensor([0.1, 0.1])
  term.std_walking = torch.tensor([0.2, 0.2])
  term.std_running = torch.tensor([0.4, 0.4])
  term.default_joint_pos = torch.zeros(len(speeds), 2)
  cfg = SceneEntityCfg("robot", joint_ids=[0, 1])
  value = term(env, {}, {}, {}, cfg, "v")
  torch.testing.assert_close(
    value[[0, 2, 4, 5]], torch.exp(-torch.tensor([1, 0.25, 0.0625, 0.0625]))
  )
  assert abs(value[1] - value[3]) < 1e-6
  assert torch.isfinite(value).all()


def test_configure_preserves_other_weights():
  cfg = NS(
    rewards={
      "lin": RewardTermCfg(
        func=r.base.track_linear_velocity,
        weight=2,
        params={"std": 0.5, "command_name": "v"},
      ),
      "pose": RewardTermCfg(func=r.base.variable_posture, weight=0.7),
      "slip": RewardTermCfg(func=r.base.feet_slip, weight=-0.2),
      "action": RewardTermCfg(func=lambda env: None, weight=-0.1),
    }
  )
  r.configure_low_rewards(cfg)
  assert cfg.rewards["lin"].weight == 0.5
  assert cfg.rewards["lin_fine"].weight == 1.5
  assert "std" not in cfg.rewards["lin_fine"].params
  assert cfg.rewards["pose"].func is r.variable_posture
  assert cfg.rewards["pose"].weight == 0.7
  assert cfg.rewards["slip"].weight == -0.2
  assert cfg.rewards["action"].weight == -0.1
