"""空视野、量程噪声和重叠记忆的反例；合成通过不代表真实通关。"""

from types import SimpleNamespace

import pytest
import torch
from support.black.causal_visual_gru import CausalVisualGRU
from support.black.depth_him import augment_depth

from black.checkpoint import initial_models
from black.course_evaluation import roster as course_roster
from black.course_evaluation import score_trial as course_score
from black.eval_protocol import roster
from black.eval_score import score_trial
from black.runtime import AlignedFrames, NativeEnv
from black.student import SharedStudent
from black.test_course_evaluation import make_trial as make_course_trial
from black.test_eval_score import make_trial


def test_empty_range_is_healthy_but_nonfinite_raw_capture_is_not():
  raw = torch.ones(6, 90, 160, 1)
  raw[0], raw[1], raw[2] = -1, 8, 0.1
  raw[3, 0, 0], raw[4, 0, 0] = float("nan"), float("inf")
  renderer = SimpleNamespace(
    width=160,
    height=90,
    camera_id=0,
    _direct160_owned=(
      torch.tensor([[16 / 9, 1.0]]),
      torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]),
    ),
    frames_for=lambda ids: [(ids, raw[ids])] if len(ids) else [],
  )
  env = object.__new__(NativeEnv)
  env.count, env.frames = 6, AlignedFrames(6, "cpu")
  env.renderer, env.training_sensors = renderer, True
  env.wp = SimpleNamespace(synchronize_device=lambda _: None)
  env.plant = SimpleNamespace(sim=SimpleNamespace(wp_device="cpu"))
  env.critic_context = lambda: torch.zeros(6, 3)
  torch.manual_seed(71)
  obs = env.bind({"history": torch.zeros(6, 225), "critic": torch.zeros(6, 446)})
  assert env.frames.valid.tolist() == [True, True, True, False, False, True]
  assert env.frames.has_hit.tolist() == [False, False, False, False, False, True]
  assert not obs["geometry"][:5].any() and obs["geometry"][5].any()
  assert not env.frames.depth[:5].any()
  assert not len(env.frames.due())
  student = SharedStudent(*initial_models())
  with torch.no_grad():
    _, _, state = student(
      torch.zeros(6, 5, 45), env.frames.depth, env.frames.metadata()
    )
  assert state.fresh_count.tolist() == [1, 1, 1, 0, 0, 1]


@pytest.mark.parametrize("distance", [0.2, 4.0])
def test_noise_does_not_create_an_extra_half_of_boundary_dropouts(distance):
  torch.manual_seed(72)
  depth = torch.full((1, 90, 160, 1), distance)
  augmented = augment_depth(depth)
  kept = augmented[augmented > 0]
  assert 0.985 < kept.numel() / depth.numel() < 0.995
  assert ((kept >= 0.2) & (kept <= 4.0)).all()
  # 原始量程外像素仍丢弃，不能夹紧成虚构的近/远表面。
  raw = torch.tensor([0.1, 4.1, -1, float("nan"), float("inf")])
  assert not augment_depth(raw).any()


@pytest.mark.parametrize("suite", ["general", "platform", "broken_bridge"])
def test_course_empty_view_can_pass_but_unhealthy_capture_cannot(suite):
  spec = course_roster(suite, 71)[0]
  trial = make_course_trial(spec)
  for row in trial["rows"]:
    row["depth_has_hit"] = False
  assert course_score(spec, trial)
  trial["rows"][20]["depth_valid"] = False
  assert not course_score(spec, trial)


def test_flat_evaluation_does_not_grant_camera_failure_a_pass():
  spec = roster("flat8", 71)[0]
  trial = make_trial(spec)
  for row in trial["rows"]:
    row["depth_has_hit"] = False
  assert score_trial(spec, trial)["status"] == "pass"
  trial["rows"][20]["depth_valid"] = False
  assert score_trial(spec, trial)["status"] == "fail"
  trial["rows"][20]["depth_valid"] = True
  trial["rows"][20]["capture_tick"] -= 5
  with pytest.raises(ValueError, match="stale"):
    score_trial(spec, trial)


def test_handoff_promotes_warm_memory_instead_of_zeroing_and_blends_gradually():
  torch.manual_seed(73)
  gru = CausalVisualGRU(enabled=True)
  visual = torch.randn(51, 1, 32)
  state = None
  for tick in range(251):
    frame = tick // 5
    row = dict(
      tick=tick,
      episode=torch.zeros(1, dtype=torch.long),
      capture_episode=torch.zeros(1, dtype=torch.long),
      frame=torch.tensor([frame]),
      capture_tick=torch.tensor([frame * 5]),
      mask=torch.tensor([True]),
    )
    if tick == 250:
      before = state
    with torch.no_grad():
      _, state = gru(visual[frame], state=state, **row)
  assert before.fresh_count.item() == 50
  assert state.fresh_count.item() == 26
  expected = gru.gru(visual[-1], before.warm_hidden)
  torch.testing.assert_close(state.hidden, expected)
  assert not torch.allclose(state.hidden, gru.gru(visual[-1], torch.zeros(1, 64)))
  torch.testing.assert_close(state.warm_hidden, gru.gru(visual[-1], torch.zeros(1, 64)))
  # 交接前最后一帧完全使用已预热25帧的状态，交接后新段只占1/25。
  with torch.no_grad():
    gru.projection.weight.normal_(std=0.1)
    row["tick"] += 1
    output, _ = gru(visual[-1], state=state, **row)
  expected_visual = visual[-1] + gru.projection(
    state.hidden * 0.96 + state.warm_hidden * 0.04
  )
  torch.testing.assert_close(output, expected_visual)


def test_overlap_window_is_bounded_and_bad_or_held_frames_do_not_advance():
  gru = CausalVisualGRU(enabled=True, memory_frames=4)
  state = None
  rows = []
  for tick in range(16):
    rows.append(
      dict(
        tick=tick,
        episode=torch.tensor([int(tick >= 12), 0]),
        capture_episode=torch.tensor([int(tick >= 12), 0]),
        frame=torch.tensor([tick, tick // 2]),
        capture_tick=torch.tensor([tick, tick - tick % 2]),
        mask=torch.tensor([tick != 10, tick != 10]),
      )
    )
  for row in rows:
    previous = None if state is None else state.detach()
    _, state = gru(torch.ones(2, 32), state=state, **row)
    assert (state.fresh_count <= 4).all()
    if row["tick"] == 10:
      torch.testing.assert_close(state.hidden, previous.hidden)
      torch.testing.assert_close(state.warm_hidden, previous.warm_hidden)
    if row["tick"] == 12:
      assert state.fresh_count[0] == 1 and not state.warm_hidden[0].any()


@pytest.mark.parametrize("window", [0, 1, 3, -2])
def test_overlap_requires_two_equal_nonempty_halves(window):
  with pytest.raises(ValueError, match="even"):
    CausalVisualGRU(enabled=True, memory_frames=window)
