"""CPU配置与时序反例；不构造物理仿真或GPU环境。"""

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from eval_adapter import EvaluationAdapter
from eval_protocol import command, padded_batch, roster
from runtime import AlignedFrames, NativeEnv, native_imports

native_imports()

from grid_course import GridCurriculum  # noqa: E402
from parameters import PARAMETERS  # noqa: E402
from passage_course import POLICY, PassageCourse  # noqa: E402
from passage_tracks import ObservedPassage  # noqa: E402
from specialty_env import make_specialty_configs  # noqa: E402


class Scene(dict):
  terrain: Any


def configured(rows):
  cfg, _ = make_specialty_configs(
    "stairs", num_envs=32, seed=73, course_parameters=PARAMETERS, passage_policy=POLICY
  )
  adapter = EvaluationAdapter(rows)
  adapter.configure(cfg)
  return adapter, cfg


def test_real_config_changes_are_private_and_command_publishes_before_history(
  monkeypatch,
):
  import specialty_env

  rows = padded_batch(roster("flat8", 73))

  class CommandStub:
    def __init__(self, cfg, env):
      self._env = env
      self.num_envs = 32
      self.device = "cpu"
      self.flat_mask = torch.zeros(32, dtype=torch.bool)
      self.vel_command_b = torch.zeros(32, 3)

    def compute(self, dt, env_ids=None):
      self.vel_command_b.fill_(9)

    def reset(self, env_ids):
      self.vel_command_b.fill_(7)

  monkeypatch.setattr(specialty_env, "SpecialtyCommand", CommandStub)
  _, cfg = configured(rows)
  untouched, _ = make_specialty_configs(
    "stairs", num_envs=32, seed=73, course_parameters=PARAMETERS, passage_policy=POLICY
  )
  assert type(untouched.commands["twist"]).__name__ == "SpecialtyCommandCfg"
  assert untouched.curriculum["specialty_course"].func.__name__ == "course_update"
  assert cfg.terminations == untouched.terminations
  scene = Scene()
  scene.terrain = SimpleNamespace(
    terrain_types=torch.zeros(32, dtype=torch.long),
    terrain_levels=torch.zeros(32, dtype=torch.long),
    terrain_origins=torch.randn(11, 7, 3),
    env_origins=torch.zeros(32, 3),
  )
  env = SimpleNamespace(scene=scene, common_step_counter=149)
  term = cfg.commands["twist"].build(env)
  term.compute(0.02)
  torch.testing.assert_close(
    term.vel_command_b, torch.tensor([command(row, 149) for row in rows])
  )
  env.common_step_counter = 150
  term.reset(torch.tensor([0, 1]))
  torch.testing.assert_close(
    term.vel_command_b, torch.tensor([command(row, 150) for row in rows])
  )
  term.compute(0.0)
  assert term.flat_mask.all()
  assert (scene.terrain.terrain_types == 0).all()
  torch.testing.assert_close(
    term.vel_command_b, torch.tensor([command(row, 150) for row in rows])
  )


def test_flat_orientation_keeps_original_yaw_range():
  cfg, _ = make_specialty_configs(
    "stairs", num_envs=32, seed=73, course_parameters=PARAMETERS, passage_policy=POLICY
  )
  calls = []
  cfg.events["reset_base"].func = lambda env, ids, pose, velocity: calls.append(
    (ids, pose, velocity)
  )
  rows = padded_batch(roster("axes", 73))
  adapter = EvaluationAdapter(rows)
  adapter.configure(cfg)
  cfg.events["reset_base"].func(
    SimpleNamespace(num_envs=32, device="cpu"),
    None,
    {"yaw": (-0.1, 0.1)},
    {"x": (-0.2, 0.2)},
  )
  assert len(calls) == 2
  assert sorted(torch.cat([row[0] for row in calls]).tolist()) == list(range(32))
  assert calls[0][1]["yaw"] == (-0.1, 0.1)
  assert calls[1][1]["yaw"] == (math.pi - 0.1, math.pi + 0.1)
  assert calls[0][2] == calls[1][2] == {"x": (-0.2, 0.2)}


def route_state(count):
  scene = Scene()
  scene.terrain = SimpleNamespace(
    terrain_types=torch.ones(count, dtype=torch.long),
    terrain_levels=torch.zeros(count, dtype=torch.long),
    terrain_origins=torch.randn(11, 7, 3),
    env_origins=torch.zeros(count, 3),
  )
  env = SimpleNamespace(
    num_envs=count, device="cpu", common_step_counter=0, scene=scene
  )
  return SimpleNamespace(
    env=env,
    grid=GridCurriculum(),
    routes=SimpleNamespace(core=SimpleNamespace(routes={}), sync=lambda ids: None),
    command=SimpleNamespace(
      entry_angle=torch.zeros(count),
      route_direction=torch.zeros(count, dtype=torch.long),
      cfg=SimpleNamespace(passage_course_policy=POLICY),
      route_sample_step=torch.full((count,), -1, dtype=torch.long),
    ),
    passage_course=PassageCourse(),
    passage_seen=ObservedPassage(count, "cpu"),
    height_level=0,
  )


def test_fixed_routes_keep_original_passage_and_grid_episode_identity():
  rows = roster("stairs", 73, heights_cm=(4, 15), angles_deg=(0,))
  adapter = EvaluationAdapter(rows)
  state = route_state(len(rows))
  ids = torch.arange(len(rows))
  adapter.assign(state, ids)
  assert state.env.scene.terrain.terrain_levels.tolist() == [0, 0, 5, 5]
  assert state.command.route_direction.tolist() == [-1, 1, -1, 1]
  assert state.passage_course.births == 4
  adapter.assign(state, ids)
  assert (
    state.passage_course.births == 4
  )  # curriculum和reset重复分配同tick不能重复出生。
  assert state.grid.record_many([(0, 0, 0, 0, -1, False)]) == 1
  state.passage_course.vote(0, 0, 0, 0, -1, "failure")
  state.passage_seen.expire(0)
  state.env.common_step_counter = 1
  adapter.assign(state, torch.tensor([0]))
  assert state.passage_seen.birth.tolist() == [1, 0, 0, 0]
  assert state.passage_course.births == 5
  assert state.passage_course.active[0] == (1, 0, 0, -1)


def observed_env(adapter, monkeypatch):
  import specialty_env

  count = len(adapter.rows)
  state = route_state(count)
  scene = state.env.scene
  scene.update(
    {
      "robot": SimpleNamespace(
        data=SimpleNamespace(
          root_link_lin_vel_w=torch.zeros(count, 3),
          root_link_ang_vel_w=torch.zeros(count, 3),
          projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        )
      ),
      "robot/imu_lin_vel": SimpleNamespace(data=torch.zeros(count, 3)),
      "robot/imu_ang_vel": SimpleNamespace(data=torch.zeros(count, 3)),
      "feet_ground_contact": SimpleNamespace(
        data=SimpleNamespace(found=torch.ones(count, 4))
      ),
    }
  )
  values = torch.zeros(count, 3)
  plant = state.env
  plant.command_manager = SimpleNamespace(get_command=lambda name: values)
  plant.reset_terminated = torch.zeros(count, dtype=torch.bool)
  plant.reset_time_outs = torch.zeros(count, dtype=torch.bool)
  term = SimpleNamespace(
    raw_action=torch.ones(count, 12),
    _processed_actions=torch.full((count, 12), 0.25),
    control_delay=SimpleNamespace(
      lags=torch.zeros(count, dtype=torch.long), history=torch.ones(count, 3, 12)
    ),
  )
  plant.action_manager = SimpleNamespace(
    action=term.raw_action, get_term=lambda name: term
  )
  state.failed = torch.zeros(count, dtype=torch.bool)
  state.finished = torch.zeros(count, dtype=torch.bool)
  state.ledger = SimpleNamespace(settled_steps=torch.zeros(count, dtype=torch.long))
  plant.reward_manager = SimpleNamespace(compute=lambda dt: torch.zeros(count))
  env = SimpleNamespace(
    plant=plant,
    frames=AlignedFrames(count, "cpu"),
    metrics=SimpleNamespace(update=lambda *args: None),
  )
  monkeypatch.setattr(specialty_env, "specialty_state", lambda env: state)
  adapter.attach(env)
  native = {"history": torch.zeros(count, 5, 45)}
  return env, state, native, values


def test_terminal_evidence_is_cloned_before_reset_and_not_next_episode(monkeypatch):
  adapter = EvaluationAdapter(roster("stairs", 73, heights_cm=(4,), angles_deg=(0,)))
  env, state, native, _ = observed_env(adapter, monkeypatch)
  adapter.begin(native)
  for name in ("track_linear_velocity", "track_angular_velocity"):
    adapter.witness(env.plant, name)
  env.plant.common_step_counter = 1
  env.plant.reset_terminated[0] = True
  state.finished[0] = True
  state.ledger.settled_steps[0] = 6
  state.passage_seen.stage[0] = 12
  state.passage_seen.landing[0] = True
  state.passage_seen.pass_once[0] = True
  flags = torch.zeros(2, 6, dtype=torch.bool)
  flags[0, 4] = True
  env.metrics.update(flags)
  env.plant.reward_manager.compute(0.02)
  state.finished.zero_()
  state.passage_seen.stage.zero_()
  flags.zero_()
  done = torch.tensor([True, False])
  env.frames.advance(done)
  adapter.finish(native, done)
  result = adapter.post()
  assert result["episode"].tolist() == [0, 0]
  assert env.frames.episode.tolist() == [1, 0]
  assert result["ordered_treads"].tolist() == [12, 0]
  assert result["strict_finished"].tolist() == [True, False]
  assert result["safe_exit"].tolist() == [True, False]
  assert result["tick"] == 0 and result["closed_tick"] == 1
  assert result["reward_command_verified"] is True
  env.plant.action_manager.action.zero_()
  assert result["action"].eq(1).all()
  assert result["raw_action"].eq(1).all()
  assert result["delayed_action"].eq(1).all()
  assert result["processed_target"].eq(0.25).all()
  adapter.detach()


def test_stale_history_and_changed_reward_command_are_rejected(monkeypatch):
  adapter = EvaluationAdapter(roster("stairs", 73, heights_cm=(4,), angles_deg=(0,)))
  env, _, native, values = observed_env(adapter, monkeypatch)
  native["history"][0, -1, 30] = 0.2
  with pytest.raises(RuntimeError, match="history"):
    adapter.begin(native)
  adapter.armed = None
  native["history"].zero_()
  adapter.begin(native)
  values[0, 0] = 0.2
  with pytest.raises(RuntimeError, match="奖励命令"):
    adapter.witness(env.plant, "track_linear_velocity")
  env.plant.common_step_counter = 1
  adapter.capture()
  with pytest.raises(RuntimeError, match="tracking"):
    adapter.finish(native, torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize("diagnostic_fails", [False, True])
def test_constructor_keeps_first_fault_when_diagnostic_or_close_fails(
  tmp_path, monkeypatch, diagnostic_fails
):
  import camera_contract
  import direct160
  import fault_capture
  import specialty_env

  import mjlab.envs
  import mjlab.rl
  from mjlab.tasks.velocity.config.black_actuation import actuation

  original = FloatingPointError("first bind fault")

  class ActionStub:
    target_names = [
      f"{leg}_{joint}_joint"
      for leg in ("FL", "FR", "RR", "RL")
      for joint in ("hip", "thigh", "calf")
    ]
    scale = 0.25
    cfg = SimpleNamespace(delay_enabled=True, clear_history_on_reset=False)

  plant = SimpleNamespace(
    max_episode_length=900,
    sim=SimpleNamespace(
      mj_model=SimpleNamespace(camera=lambda name: SimpleNamespace(id=0)),
      wp_model=object(),
      wp_data=object(),
    ),
    action_manager=SimpleNamespace(get_term=lambda name: ActionStub()),
  )
  monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", lambda *args, **kwargs: plant)
  monkeypatch.setattr(
    mjlab.rl,
    "RslRlVecEnvWrapper",
    lambda *args, **kwargs: SimpleNamespace(get_observations=lambda: {}),
  )
  monkeypatch.setattr(actuation, "BlackDelayedJointPositionAction", ActionStub)
  monkeypatch.setattr(camera_contract, "verify_camera", lambda *args: {})
  monkeypatch.setattr(direct160, "make_renderer", lambda **kwargs: (object(), None, {}))
  monkeypatch.setattr(
    specialty_env,
    "specialty_state",
    lambda env: SimpleNamespace(
      episode_metrics=SimpleNamespace(configure=lambda *args: None)
    ),
  )
  monkeypatch.setattr(
    fault_capture,
    "attach",
    lambda *args, **kwargs: SimpleNamespace(fault={"first": torch.tensor([1])}),
  )

  def fail_bind(*args):
    raise original

  def fail_close(*args):
    raise OSError("close fault")

  monkeypatch.setattr(NativeEnv, "bind", fail_bind)
  monkeypatch.setattr(NativeEnv, "close", fail_close)
  if diagnostic_fails:

    def fail_write(*args):
      raise OSError("diagnostic fault")

    monkeypatch.setattr(fault_capture, "write_fault", fail_write)
  with pytest.raises(FloatingPointError) as caught:
    NativeEnv(32, 73, "cpu", tmp_path)
  assert caught.value is original
  details = original.__dict__["failure_details"]
  assert details["close_error"] == "OSError('close fault')"
  if diagnostic_fails:
    assert details["diagnostic_error"] == "OSError('diagnostic fault')"
  else:
    saved = torch.load(tmp_path / "first_fault/FIRST_BAD.pt", weights_only=True)
    assert saved["first"].tolist() == [1]
