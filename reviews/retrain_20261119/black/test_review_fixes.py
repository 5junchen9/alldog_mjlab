"""具体反例：奖励方向、记忆边界、深度范围与故障退出。"""

from types import SimpleNamespace

import pytest
import torch
from support.black.config import TASKS, configure_ppo_schedule, make_general_configs
from support.black.exit_stability import ExitStabilityLedger
from support.black.parameters import PARAMETERS
from support.black.specialty_env import make_specialty_configs

from black.checkpoint import initial_models
from black.curriculum import capture_curriculum, restore_curriculum
from black.fault_capture import check_control_boundary
from black.run import update_him
from black.student import SharedStudent
from black.student_training import EpisodePrefix, _prefix_memory
from black.terrain import geometry, reduce_depth


def test_all_six_specialties_compensate_completion_and_shape_all_49_hold_steps():
  for task in TASKS:
    cfg, _ = make_specialty_configs(task, num_envs=32, course_parameters=PARAMETERS)
    assert {
      "specialty_completion_value",
      "specialty_progress_dense_cost",
      "specialty_exit_stability",
    } <= cfg.rewards.keys()
    assert cfg.rewards["foot_clearance"].params["penalize_overshoot"] is False
  ledger = ExitStabilityLedger(1, "cpu", settle_steps=50)
  total = 0.0
  for tick in range(1, 51):
    ledger.prepare(
      tick,
      0.02,
      original_stable_steps=torch.tensor([tick]),
      active=torch.tensor([True]),
      failed=torch.tensor([False]),
      valid=torch.tensor([True]),
    )
    reward = ledger.consume(tick)
    if tick <= 49:
      assert reward.item() > 0
    total += reward.item() * 0.02
    assert ledger.consume(tick).item() == 0  # 同步重复读取不重复付款。
  assert total == pytest.approx(1.0)


@pytest.mark.parametrize(
  "course,lr",
  [
    ("flat_foundation", 1e-3),
    ("flat_axes", 1e-3),
    ("general", 3e-4),
    ("specialty", 1e-4),
  ],
)
def test_scratch_stage_learning_rates(course, lr):
  _, rl = make_general_configs(32)
  configure_ppo_schedule(rl, course)
  assert rl.algorithm.learning_rate == lr and rl.algorithm.schedule == "adaptive"


def test_him_command_changes_cannot_change_dynamics_encoding_and_tail_is_balanced(
  monkeypatch,
):
  teacher, _, _ = initial_models()
  h = torch.randn(5, 5, 45)
  changed = h.clone()
  changed[..., 30:33] += 20
  torch.testing.assert_close(teacher.him(h), teacher.him(changed), rtol=0, atol=0)
  seen = []
  original = teacher.him.estimator_losses

  def losses(history, frame, velocity):
    seen.append(len(history))
    return original(history, frame, velocity)

  monkeypatch.setattr(teacher.him, "estimator_losses", losses)
  optimizer = torch.optim.Adam(teacher.him.parameters(), lr=1e-4)
  update_him(
    teacher.him,
    optimizer,
    [(torch.randn(2049, 5, 45), torch.randn(2049, 45), torch.randn(2049, 3))],
    "cpu",
  )
  assert seen == [1025, 1024]
  assert all(p.grad is None for p in teacher.him.parameters())


def test_overlapping_segment_cache_matches_live_gru_after_new_weights(monkeypatch):
  torch.set_num_threads(1)
  student = SharedStudent(*initial_models())
  count, frames = 2, 107
  histories = torch.randn(frames, count, 5, 45)
  depths = torch.rand(frames, count, 48, 64, 1) + 0.5
  rows = [
    dict(
      tick=i * 5,
      episode=torch.zeros(count, dtype=torch.long),
      capture_episode=torch.zeros(count, dtype=torch.long),
      frame=torch.full((count,), i, dtype=torch.long),
      capture_tick=torch.full((count,), i * 5, dtype=torch.long),
      mask=torch.ones(count, dtype=torch.bool),
    )
    for i in range(frames)
  ]
  # 连续控制步重放；只在5步倍数处采新帧。
  expanded_rows, expanded_depth, expanded_h = [], [], []
  for tick in range((frames - 1) * 5 + 1):
    i = tick // 5
    expanded_rows.append({**rows[i], "tick": tick})
    expanded_depth.append(depths[i])
    expanded_h.append(histories[i])
  sample = dict(
    history=torch.stack(expanded_h),
    depth=torch.stack(expanded_depth),
    metadata=expanded_rows,
  )
  with torch.no_grad():
    _, _, state = student.forward_sequence(
      sample["history"], sample["depth"], sample["metadata"]
    )
  cache = EpisodePrefix(count, 6000, student.gru.max_age)
  cache.extend(sample)
  assert cache.length.tolist() == [32, 32]
  assert state.fresh_count.tolist() == [32, 32]
  with torch.no_grad():
    student.depth_encoder[0].weight.add_(0.01)
    _, _, reference = student.forward_sequence(
      sample["history"], sample["depth"], sample["metadata"]
    )
  refreshed = cache.refresh(student, state, "cpu")
  torch.testing.assert_close(refreshed.hidden, reference.hidden, rtol=1e-6, atol=1e-7)
  torch.testing.assert_close(
    refreshed.warm_hidden, reference.warm_hidden, rtol=1e-6, atol=1e-7
  )
  monkeypatch.setattr(
    "black.student_training.Path.read_text", lambda _: "MemAvailable: 67108864 kB\n"
  )
  capacity, size, _ = _prefix_memory(4096, 6000)
  assert capacity == 50 and size < 3 * 1024**3


def test_range_filter_preserves_real_ray_and_rejects_near_and_far_background():
  raw = torch.full((1, 90, 160, 1), 100.0)
  raw[0, 0, 0] = 0.01
  raw[0, 1, 1] = 0.5
  d, pixels = reduce_depth(raw)
  points = geometry(d, pixels, torch.tensor([-1.0, 1.0, -1.0, 1.0])).reshape(-1, 4)
  assert (points[:, 3] > 0).any()
  assert points[points[:, 3] > 0, 2].eq(0.125).all()
  assert torch.isfinite(points).all()


def test_boundary_guard_finds_hidden_warmstart_nan_without_observation_nan():
  import warp as wp

  wp.init()
  values = {
    name: torch.zeros(3, 2)
    for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl")
  }
  data = SimpleNamespace(
    **{name: wp.from_torch(value) for name, value in values.items()}
  )
  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    sim=SimpleNamespace(wp_data=data, wp_device=wp.get_device("cpu")),
  )
  check_control_boundary(env)
  values["qacc_warmstart"][2, 0] = float("nan")
  with pytest.raises(FloatingPointError) as caught:
    check_control_boundary(env)
  assert caught.value.failure_details["bad_environment_ids"] == [2]
  assert caught.value.failure_details["first_bad_substep_captured"] is False


def test_delay_reset_clears_only_the_reset_lane():
  from mjlab.tasks.velocity.config.black_actuation.actuation import ControlStepDelay

  delay = ControlStepDelay(2, 12, "cpu")
  delay.lags.fill_(2)
  for i in (1, 2, 3):
    delay.push(torch.full((2, 12), float(i)))
  delay.reset(torch.tensor([0]), clear_history=True)
  delay.lags.fill_(2)
  result = delay.push(torch.full((2, 12), 4.0))
  assert result[0].eq(0).all() and result[1].eq(2).all()


def test_general_curriculum_roundtrip_preserves_closed_windows_and_levels():
  from black.curriculum import GeneralProgress
  from mjlab.tasks.velocity.config.black_rough.mdp import FailureState

  st = object.__new__(FailureState)
  st.type_ceilings = [0, 2, 1, 1, 1, 1, 1]
  st.histories = [[True] * 5 for _ in range(7)]
  plant = SimpleNamespace(
    common_step_counter=200,
    black_rough_state=st,
    general_progress=GeneralProgress(),
    scene=SimpleNamespace(terrain=SimpleNamespace(terrain_levels=torch.tensor([0, 2]))),
  )
  plant.general_progress.windows[1, 1] = [True] * 32
  env = SimpleNamespace(plant=plant, course="general", device="cpu")
  saved = capture_curriculum(env)
  st.type_ceilings[1] = 1
  plant.general_progress.windows.clear()
  restore_curriculum(env, saved)
  assert st.type_ceilings[1] == 2
  assert len(plant.general_progress.windows[1, 1]) == 32


def test_gap_miss_does_not_poison_clearance_or_next_landing_peak(monkeypatch):
  from mjlab.tasks.velocity.mdp import rewards

  class Height:
    num_frames = 4
    data = SimpleNamespace(heights=torch.full((1, 4), 0.1))
    support_ray_hit = torch.ones(1, 4, dtype=torch.bool)

  monkeypatch.setattr(rewards, "TerrainHeightSensor", Height)
  height = Height()
  contact = SimpleNamespace(data=SimpleNamespace(found=torch.zeros(1, 4)))
  contact.compute_first_contact = lambda dt: contact.data.found.bool()
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=0.02,
    extras={"log": {}},
    command_manager=SimpleNamespace(
      get_command=lambda _: torch.tensor([[0.2, 0.0, 0.0]])
    ),
    scene={
      "height": height,
      "contact": contact,
      "robot": SimpleNamespace(
        data=SimpleNamespace(site_lin_vel_w=torch.ones(1, 4, 3))
      ),
    },
  )
  params = dict(
    height_sensor_name="height",
    sensor_name="contact",
    target_height=0.1,
    command_name="twist",
    command_threshold=0.01,
  )
  peak = rewards.feet_swing_height(SimpleNamespace(params=params), env)
  peak(env, **params)
  height.data.heights.fill_(1.0)
  height.support_ray_hit.fill_(False)
  assert peak(env, **params).item() == 0.0
  asset = SimpleNamespace(name="robot", site_ids=[0, 1, 2, 3])
  assert rewards.feet_clearance(env, 0.1, "height", asset_cfg=asset).item() == 0.0
  contact.data.found.fill_(1)
  height.support_ray_hit.fill_(True)
  height.data.heights.zero_()
  assert peak(env, **params).item() == pytest.approx(0.0)


def test_course_privilege_is_available_to_critic_without_changing_actor():
  from tensordict import TensorDict

  from black.ppo import build_ppo

  teacher, _, _ = initial_models()
  obs = TensorDict(
    dict(
      history=torch.zeros(4, 225),
      geometry=torch.zeros(4, 768),
      critic=torch.zeros(4, 446),
      course_state=torch.zeros(4, 3),
    ),
    [4],
  )
  ppo = build_ppo(obs, teacher)
  assert ppo.critic.obs_dim == 449
  with torch.no_grad():
    original = ppo.actor(obs, stochastic_output=False)
    obs["course_state"][:] = 1.0
    changed = ppo.actor(obs, stochastic_output=False)
  torch.testing.assert_close(original, changed, rtol=0, atol=0)


def test_cold_curriculum_restart_before_first_physics_step():
  from black.runtime import NativeEnv

  env = object.__new__(NativeEnv)
  env.count, env.device = 2, "cpu"
  plant = SimpleNamespace(
    episode_length_buf=torch.ones(2, dtype=torch.long),
    termination_manager=SimpleNamespace(
      terminated=torch.ones(2, dtype=torch.bool),
      time_outs=torch.ones(2, dtype=torch.bool),
    ),
    reset=lambda: ({"ready": torch.ones(2)}, {}),
  )
  env.plant = plant
  env.bind = lambda native: native
  assert not hasattr(plant, "reset_terminated")
  env.restart_after_curriculum(99)
  assert plant.common_step_counter == 100
  assert not plant.reset_terminated.any() and not plant.reset_time_outs.any()
  assert env.frames.episode.eq(0).all()


def test_general_curriculum_initial_reset_does_not_read_post_step_aliases(monkeypatch):
  from black.curriculum import configure
  from mjlab.tasks.velocity.config.black_rough import mdp

  cfg, _ = make_general_configs(32)
  cfg.curriculum["rough_levels"].func = lambda env, ids: {}
  configure(cfg, "general")
  st = SimpleNamespace(
    traversal=torch.zeros(32, dtype=torch.bool),
    failed=torch.zeros(32, dtype=torch.bool),
    type_ceilings=[0, 1, 1, 1, 1, 1, 1],
  )
  monkeypatch.setattr(mdp, "state", lambda _: st)
  env = SimpleNamespace(
    num_envs=32,
    device="cpu",
    episode_length_buf=torch.zeros(32, dtype=torch.long),
    command_manager=SimpleNamespace(
      get_term=lambda _: SimpleNamespace(flat_mask=torch.zeros(32, dtype=torch.bool))
    ),
    termination_manager=SimpleNamespace(
      terminated=torch.zeros(32, dtype=torch.bool),
      time_outs=torch.zeros(32, dtype=torch.bool),
    ),
    scene=SimpleNamespace(
      terrain=SimpleNamespace(
        terrain_types=torch.ones(32, dtype=torch.long),
        terrain_levels=torch.ones(32, dtype=torch.long),
      )
    ),
  )
  assert not hasattr(env, "reset_terminated")
  cfg.curriculum["rough_levels"].func(env, torch.arange(32))
  assert env.general_progress.windows == {}


def test_external_reset_metrics_accept_full_hold_reward_without_counter_precision_loss(
  tmp_path,
):
  import json

  from support.black.stair_metrics import EpisodeMetrics

  zero = torch.zeros(1, dtype=torch.long)
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    common_step_counter=2**25 + 3,
    step_dt=0.02,
    scene=SimpleNamespace(terrain=SimpleNamespace(terrain_levels=zero)),
    black_rough_state=SimpleNamespace(reason=torch.zeros(1, 4, dtype=torch.bool)),
  )
  st = SimpleNamespace(
    command=SimpleNamespace(
      entry_angle=torch.zeros(1),
      route_direction=torch.ones(1, dtype=torch.long),
      flat_mask=torch.zeros(1, dtype=torch.bool),
    ),
    failed=torch.zeros(1, dtype=torch.bool),
    finished=torch.zeros(1, dtype=torch.bool),
    ledger=SimpleNamespace(paid_ticks=zero),
    stair_ledger=SimpleNamespace(paid_milli=zero, max_stage=zero),
    exit_stability=SimpleNamespace(paid_total=torch.tensor([0.5])),
    completion_value=SimpleNamespace(paid=torch.zeros(1, dtype=torch.bool)),
  )
  metrics = EpisodeMetrics(env)
  metrics.configure(tmp_path, {"source_actual_updates": 0})
  metrics.active.fill_(True)
  metrics.birth[:] = 2**25 + 1
  metrics.reset(torch.tensor([0]), st)
  end = json.loads(metrics.path.read_text().splitlines()[0])
  assert end["exit_stability_event_units"] == 0.5
  assert end["birth_step"] == 2**25 + 1
  assert end["terminal_step"] == 2**25 + 3
  assert end["outcome"] == "censored"
