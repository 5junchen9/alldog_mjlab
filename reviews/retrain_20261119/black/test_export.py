"""独立TorchScript加载与合成相机时序；不代表硬件部署验收。"""

import pytest
import torch

from black.checkpoint import CONTRACT
from black.export import export_student, load_student_export, teacher_inference_module
from black.model import Teacher, make_modules
from black.student import SharedStudent


@pytest.fixture
def student():
  threads = torch.get_num_threads()
  torch.set_num_threads(1)
  torch.manual_seed(61)
  him, actor, cnn, gru = make_modules()
  with torch.no_grad():
    gru.projection.weight.normal_(std=0.03)
  yield SharedStudent(Teacher(him, actor), cnn, gru).eval()
  torch.set_num_threads(threads)


def test_independent_student_jit_matches_fresh_held_invalid_and_reset(
  student, tmp_path
):
  path = tmp_path / "student_jit.pt"
  manifest = export_student(student, path)
  loaded, readback = load_student_export(path, CONTRACT["action_order"])
  assert readback == manifest and not manifest["physical_deployment_validated"]
  state = None
  hidden = torch.zeros(2, 64)
  count = torch.zeros(2, dtype=torch.long)
  old_episode = torch.full((2,), -1)
  old_frame = torch.full((2,), -1)
  for tick in range(265):
    episode = torch.tensor([int(tick >= 8), 0])
    birth = torch.tensor([8 if tick >= 8 else 0, 0])
    capture = birth + ((tick - birth) // 5) * 5
    frame = capture * 2 + torch.arange(2)
    mask = torch.tensor([True, tick < 7])
    row = dict(
      tick=tick,
      episode=episode,
      capture_episode=episode,
      frame=frame,
      capture_tick=capture,
      mask=mask,
    )
    history = torch.randn(2, 5, 45)
    depth = torch.full((2, 48, 64, 1), 1.0 + tick / 100)
    depth[1, 0, 0] = float("nan")
    reset = episode != old_episode
    fresh = mask & ((frame != old_frame) | reset)
    with torch.no_grad():
      expected, _, state = student(history, depth, row, state)
      action, hidden, count = loaded(
        history, depth, hidden, fresh, reset, mask, tick - capture, count
      )
    torch.testing.assert_close(action, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden, state.hidden, rtol=0, atol=0)
    torch.testing.assert_close(count, state.fresh_count)
    old_episode, old_frame = episode, state.frame


def test_export_rejects_other_rear_leg_order_before_loading(student, tmp_path):
  path = tmp_path / "student_jit.pt"
  export_student(student, path)
  other = [
    f"{leg}_{joint}"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
  ]
  with pytest.raises(ValueError, match="action order"):
    load_student_export(path, other)
  with pytest.raises(FileExistsError):
    export_student(student, path)


def test_teacher_jit_uses_the_same_him_geometry_and_actor(student, tmp_path):
  history, geometry = torch.randn(3, 5, 45), torch.randn(3, 768)
  module = torch.jit.script(teacher_inference_module(student.teacher))
  path = tmp_path / "teacher_jit.pt"
  module.save(str(path))
  with torch.no_grad():
    actual = torch.jit.load(str(path))(history, geometry)
    expected = student.teacher(history, geometry)
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_serialization_failure_does_not_publish_partial_model(
  student, tmp_path, monkeypatch
):
  def fail(_module, stream):
    stream.write(b"incomplete")
    raise OSError("injected export write failure")

  monkeypatch.setattr(torch.jit, "save", fail)
  path = tmp_path / "student_jit.pt"
  with pytest.raises(OSError, match="injected"):
    export_student(student, path)
  assert not list(tmp_path.iterdir())
