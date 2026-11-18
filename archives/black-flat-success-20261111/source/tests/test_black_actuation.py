"""CPU tensor tests; callable directly without installing pytest in the runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import torch

from mjlab.actuator import ActuatorCmd, FusedActuatorGroup
from mjlab.asset_zoo.robots.black_stock.black_constants import get_black_robot_cfg
from mjlab.entity import Entity
from mjlab.tasks.velocity.config.black_actuation.actuation import (
  BlackDelayedJointPositionActionCfg,
  BlackFrictionActuator,
  BlackFrictionActuatorCfg,
  ControlStepDelay,
)

ROOT = Path(__file__).resolve().parents[1]


def _params(shape=(3, 12)):
  return {
    key: torch.full(shape, value)
    for key, value in (
      ("stiffness", 40.0),
      ("damping", 1.2),
      ("force_limit", 20.0),
      ("friction_scale", 1.0),
    )
  }


def _source_class():
  """Run only inspected source methods, with an inert IsaacGym step harness."""
  path = ROOT / "preparation/actuation/source/black_env.py"
  module = ast.parse(path.read_text())
  cls = next(n for n in module.body if isinstance(n, ast.ClassDef))
  cls.body = [
    n
    for n in cls.body
    if isinstance(n, ast.FunctionDef)
    and n.name in ("_compute_torques", "step", "reset_idx")
  ]
  inert_base = type("LeggedRobot", (), {"reset_idx": lambda self, ids: None})
  namespace = {
    "torch": torch,
    "LeggedRobot": inert_base,
    "gymtorch": SimpleNamespace(unwrap_tensor=lambda x: x),
  }
  exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
  return namespace["BlackEnv"]


def test_friction_matches_source_and_saturates_after_subtraction():
  torch.manual_seed(42)
  src = _source_class()()
  src.cfg = SimpleNamespace(
    control=SimpleNamespace(action_scale=0.25, hip_reduction=1.0, control_type="P")
  )
  src.default_dof_pos = torch.randn(3, 12)
  src.dof_pos = torch.randn(3, 12)
  src.dof_vel = torch.randn(3, 12) * 7
  src.p_gains, src.d_gains = 40.0, 1.2
  src.Kp_factors, src.Kd_factors = 1.0, 1.0
  src.torque_limits = torch.full((12,), 20.0)
  actions = torch.randn(3, 12)
  expected = src._compute_torques(actions)
  cmd = ActuatorCmd(
    src.default_dof_pos + 0.25 * actions,
    torch.zeros(3, 12),
    torch.zeros(3, 12),
    src.dof_pos,
    src.dof_vel,
  )
  actual = BlackFrictionActuator.control_law(_params(), cmd)
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  assert actual.abs().max() <= 20
  # Wrong clamp-before-friction would yield ~18.65, not 20.
  cmd = ActuatorCmd(*[torch.zeros(1, 1) for _ in range(5)])
  cmd.effort_target.fill_(100)
  cmd.vel.fill_(10)
  assert BlackFrictionActuator.control_law(_params((1, 1)), cmd).item() == 20
  cmd.effort_target.fill_(-100)
  cmd.vel.fill_(-10)
  assert BlackFrictionActuator.control_law(_params((1, 1)), cmd).item() == -20


def test_friction_opposes_motion_and_can_be_disabled():
  velocity = torch.tensor([[-1.0, -1e-5, 0.0, 1e-5, 1.0]])
  zeros = torch.zeros_like(velocity)
  cmd = ActuatorCmd(zeros, zeros, zeros, zeros, velocity)
  params = _params(velocity.shape)
  params["stiffness"].zero_()
  params["damping"].zero_()
  torque = BlackFrictionActuator.control_law(params, cmd)
  assert torch.all(torque * velocity <= 0)
  assert torque[0, 2] == 0
  torch.testing.assert_close(torque, -torch.flip(torque, dims=[1]))
  params["friction_scale"].zero_()
  torch.testing.assert_close(BlackFrictionActuator.control_law(params, cmd), zeros)


def test_delay_zero_one_two_control_steps_and_disable():
  delay = ControlStepDelay(3, 2, "cpu")
  delay.lags[:] = torch.tensor([0, 1, 2])
  for step in range(1, 6):
    actions = torch.full((3, 2), float(step))
    actual = delay.push(actions)
    expected = torch.tensor(
      [[step] * 2, [max(0, step - 1)] * 2, [max(0, step - 2)] * 2], dtype=torch.float
    )
    torch.testing.assert_close(actual, expected)
  current = torch.full((3, 2), 17.0)
  torch.testing.assert_close(delay.push(current, enabled=False), current)


def test_reset_samples_only_selected_envs_and_matches_source():
  delay = ControlStepDelay(1000, 2, "cpu")
  delay.history.fill_(9)
  ids = torch.arange(0, 1000, 2)
  torch.manual_seed(42)
  delay.reset(ids)
  assert set(delay.lags[ids].tolist()) == {0, 1, 2}
  assert torch.all(delay.lags[1::2] == 0)
  assert torch.all(delay.history == 9)  # Pinned source carries across episodes.
  source = _source_class()()
  source.cfg = SimpleNamespace(domain_rand=SimpleNamespace(delay=True, lag_timesteps=3))
  source.device = "cpu"
  source.lag_buffer = torch.zeros(1000, dtype=torch.long)
  source.last_impact_contacts = torch.zeros(1000, 4, dtype=torch.bool)
  source.stuck_time = torch.ones(1000)
  torch.manual_seed(42)
  source.reset_idx(ids)
  torch.testing.assert_close(source.lag_buffer, delay.lags)
  before = delay.lags.clone()
  for _ in range(8):
    delay.push(torch.zeros(1000, 2))
  torch.testing.assert_close(delay.lags, before)
  delay.history.fill_(9)
  delay.reset(ids, clear_history=True)
  assert torch.all(delay.history[ids] == 0)
  assert torch.all(delay.history[1::2] == 9)
  delay.reset(torch.tensor([], dtype=torch.long))
  delay.reset(slice(2, 4))
  delay.reset()


def _action_fixture(delay_enabled=True, clip=None):
  outputs = []
  robot: Any = SimpleNamespace(
    find_joints_by_actuator_names=lambda names: ([0, 1], ["a", "b"]),
    data=SimpleNamespace(
      default_joint_pos=torch.tensor([[0.8, -1.5]]).repeat(3, 1),
      encoder_bias=torch.full((3, 2), 0.01),
    ),
    set_joint_position_target=lambda value, joint_ids: outputs.append(value.clone()),
  )
  env: Any = SimpleNamespace(scene={"robot": robot}, num_envs=3, device="cpu")
  cfg = BlackDelayedJointPositionActionCfg(
    entity_name="robot",
    actuator_names=("a", "b"),
    scale=0.25,
    delay_enabled=delay_enabled,
    clip=clip,
  )
  return cfg.build(cast(Any, env)), robot, outputs


def test_action_applies_once_per_control_step_and_preserves_raw_history():
  action, robot, outputs = _action_fixture()
  action.control_delay.lags[:] = torch.tensor([0, 1, 2])
  source = _source_class()()
  source.cfg = SimpleNamespace(
    normalization=SimpleNamespace(clip_actions=100.0, clip_observations=100.0),
    domain_rand=SimpleNamespace(delay=True),
    control=SimpleNamespace(decimation=4),
  )
  source.device, source.num_envs, source.sim = "cpu", 3, None
  source.action_queue = torch.zeros(3, 4, 2)
  source.lag_buffer = torch.tensor([0, 1, 2])
  source.torques = torch.zeros(3, 2)
  source.obs_buf, source.privileged_obs_buf = torch.zeros(3, 2), None
  source.rew_buf, source.reset_buf, source.extras = None, None, {}
  source.render = lambda: None
  source.post_physics_step = lambda: (None, None)
  source.gym = SimpleNamespace(
    **{
      name: lambda *args: None
      for name in (
        "set_dof_actuation_force_tensor",
        "simulate",
        "fetch_results",
        "refresh_dof_state_tensor",
      )
    }
  )
  seen = []
  source._compute_torques = lambda value: seen.append(value.clone()) or value
  for step in range(1, 5):
    actions = torch.full((3, 2), float(step))
    source.step(actions)
    action.process_actions(actions)
    history = action.control_delay.history.clone()
    for _ in range(4):
      action.apply_actions()
      torch.testing.assert_close(
        outputs[-1], robot.data.default_joint_pos + seen[-1] * 0.25 - 0.01
      )
      torch.testing.assert_close(action.control_delay.history, history)
    torch.testing.assert_close(action.raw_action, actions)
    assert all(torch.equal(value, seen[-1]) for value in seen[-4:])
  torch.testing.assert_close(action.control_delay.history, source.action_queue)


def test_disabled_action_is_stock_affine_and_clip():
  action, _, outputs = _action_fixture(delay_enabled=False, clip={".*": (-1.0, 1.0)})
  action.control_delay.lags.fill_(2)
  actions = torch.full((3, 2), 20.0)
  action.process_actions(actions)
  action.apply_actions()
  torch.testing.assert_close(outputs[-1], torch.full((3, 2), 0.99))
  torch.testing.assert_close(action.raw_action, actions)


def test_actuator_initialization_and_fused_path():
  cfg = BlackFrictionActuatorCfg(
    target_names_expr=("a", "b"), stiffness=40.0, damping=1.2, effort_limit=20.0
  )
  entity: Any = SimpleNamespace(indexing=SimpleNamespace(ctrl_ids=torch.tensor([0, 1])))
  actuator = cfg.build(cast(Any, entity), [0, 1], ["a", "b"])
  actuator._mjs_actuators = [SimpleNamespace(id=i) for i in range(2)]  # type: ignore
  data: Any = SimpleNamespace(nworld=3)
  actuator.initialize(None, None, data, "cpu")  # type: ignore
  fused, remaining = FusedActuatorGroup.process([actuator])
  assert not remaining
  fused.initialize(3, "cpu")
  result = []
  data = SimpleNamespace(
    joint_pos_target=torch.full((3, 2), 0.2),
    joint_vel_target=torch.zeros(3, 2),
    joint_effort_target=torch.zeros(3, 2),
    joint_pos=torch.zeros(3, 2),
    joint_vel=torch.ones(3, 2),
    write_ctrl=lambda t, ids: result.append(t.clone()),
  )
  fused.apply_controls(cast(Any, data))
  expected = 8.0 - 1.2 - 0.35 * torch.tanh(torch.tensor(3.0)) - 0.1
  torch.testing.assert_close(result[-1], torch.full((3, 2), float(expected)))
  assert not actuator.has_delay  # Avoid accidental second physics-step delay.
  actuator.reset()


def test_black_imu_binding_uses_source_frame_and_local_measurements():

  model = Entity(get_black_robot_cfg()).compile()
  site = model.site("imu")
  assert model.body(int(site.bodyid[0])).name == "imu_link"
  np.testing.assert_array_equal(site.pos, np.zeros(3))
  np.testing.assert_array_equal(site.quat, [1, 0, 0, 0])
  types = {
    "imu_ang_vel": mujoco.mjtSensor.mjSENS_GYRO,
    "imu_lin_vel": mujoco.mjtSensor.mjSENS_VELOCIMETER,
    "imu_lin_acc": mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    "imu_upvector": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
  }
  for name, expected in types.items():
    assert model.sensor(name).type == expected
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.qvel[:6] = [1, 2, 3, 4, 5, 6]
  mujoco.mj_forward(model, data)
  np.testing.assert_allclose(data.sensor("imu_lin_vel").data, [1, 2, 3], atol=1e-12)
  np.testing.assert_allclose(data.sensor("imu_ang_vel").data, [4, 5, 6], atol=1e-12)
  np.testing.assert_allclose(data.sensor("imu_upvector").data, [0, 0, 1], atol=1e-12)
