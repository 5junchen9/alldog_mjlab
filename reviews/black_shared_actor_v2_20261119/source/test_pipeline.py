"""跨模块的权重移交、相机帧和reset边界测试。"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint
from runtime import AlignedFrames


def test_new_bundle_is_independent_of_r800_file(tmp_path, monkeypatch):
  teacher, cnn, gru = checkpoint.initial_models()
  history = torch.randn(2, 5, 45)
  geometry = torch.randn(2, 768)
  expected = teacher(history, geometry).detach()
  path = tmp_path / "teacher.pt"
  identity = checkpoint.save_bundle(path, teacher, cnn, gru, phase="probe", updates=0)
  actual_load = torch.load

  def bundle_only(source, *args, **kwargs):
    assert source == path
    return actual_load(source, *args, **kwargs)

  monkeypatch.setattr(torch, "load", bundle_only)
  restored, _, _, packet = checkpoint.load_bundle(path)
  torch.testing.assert_close(restored(history, geometry), expected, rtol=0, atol=0)
  assert checkpoint.teacher_identity(restored) == identity == packet["teacher_id"]
  packet["teacher"]["actor.0.bias"][0] += 1
  torch.save(packet, path)
  with pytest.raises(ValueError, match="权重身份"):
    checkpoint.load_bundle(path)


def test_frames_hold_once_and_reset_only_finished_lanes():
  frames = AlignedFrames(2, "cpu")
  ids = frames.due()
  depth = torch.ones(2, 48, 64, 1)
  points = torch.ones(2, 768)
  frames.capture(ids, depth, points)
  assert not len(frames.due())  # 同tick再次取观测不能重采样。
  first = frames.frame.clone()
  frames.advance(torch.tensor([False, True]))
  assert frames.due().tolist() == [1]
  assert frames.episode.tolist() == [0, 1]
  assert frames.frame[0] == first[0]
  assert frames.depth[0].eq(1).all() and frames.depth[1].eq(0).all()
  frames.capture(torch.tensor([1]), depth[1:], points[1:])
  for _ in range(4):
    frames.advance(torch.zeros(2, dtype=torch.bool))
  assert frames.tick == 5 and frames.due().tolist() == [0]
  capture_tick = frames.metadata()["capture_tick"]
  assert isinstance(capture_tick, torch.Tensor)
  assert capture_tick.tolist() == [0, 1]


def test_unqualified_teacher_cannot_start_student_training(tmp_path):
  with pytest.raises(ValueError, match="独立能力评估"):
    checkpoint.require_teacher_evaluation(None, "teacher")
  report = tmp_path / "evaluation.json"
  report.write_text('{"teacher_id":"other","passed":{"flat":true}}')
  with pytest.raises(ValueError):
    checkpoint.require_teacher_evaluation(report, "teacher")


def test_true_terminal_wins_over_simultaneous_timeout():
  from test_terminal_transition import fake_env

  env, _, _ = fake_env()
  _, _, returned_done, info = env.step(torch.zeros(3, 12))
  assert info["time_outs"].tolist() == [False, True, False]
  assert returned_done.tolist() == [1, 1, 0]


@pytest.mark.parametrize("complete", [False, True])
def test_manual_pass_summary_cannot_open_student_training(tmp_path, complete):
  path = tmp_path / "manual_summary.json"
  path.write_text(
    json.dumps(
      {
        "teacher_id": "teacher",
        "complete": complete,
        "action_mode": "mean",
        "trials": [],
        "passed": dict(flat=True, stairs=True, low_speed_turn=True, reverse=True),
      }
    )
  )
  with pytest.raises(ValueError):
    checkpoint.require_teacher_evaluation(path, "teacher")
