"""入口验收与教师真优化器测试；环境运动为合成CPU数据。"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import run
import torch
from model import Teacher, make_modules
from tensordict import TensorDict


@pytest.mark.parametrize("case", ["identity_mismatch", "missing_reverse"])
@pytest.mark.parametrize("source", ["--bundle", "--resume-learner"])
def test_student_gate_rejects_before_creating_environment(
  tmp_path, monkeypatch, case, source
):
  him, actor, cnn, gru = make_modules()
  teacher = Teacher(him, actor)
  report = {
    "teacher_id": run.teacher_identity(teacher),
    "passed": dict(flat=True, stairs=True, low_speed_turn=True, reverse=True),
  }
  if case == "identity_mismatch":
    report["teacher_id"] = "another_teacher"
  else:
    del report["passed"]["reverse"]
  path = tmp_path / "evaluation.json"
  path.write_text(json.dumps(report))
  monkeypatch.setattr(run, "load_bundle", lambda *_: (teacher, cnn, gru, {}))

  def forbidden(*args):
    pytest.fail("未验收教师不应创建环境或学生优化器")

  monkeypatch.setattr(run, "NativeEnv", forbidden)
  monkeypatch.setattr(run, "train_student", forbidden)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "run.py",
      "--mode",
      "student",
      "--execute",
      source,
      "selected.pt",
      "--teacher-evaluation",
      str(path),
      "--rounds",
      "1",
      "--device",
      "cpu",
      "--out",
      str(tmp_path / "rejected"),
    ],
  )
  with pytest.raises(ValueError, match="验收未通过"):
    run.main()
  assert (tmp_path / "rejected/FAILED.json").is_file()


@dataclass
class Options:
  num_learning_epochs: int = 1
  num_mini_batches: int = 2
  schedule: str = "fixed"
  learning_rate: float = 1e-4


class SyntheticEnv:
  device = "cpu"
  count = 4

  def __init__(self):
    self.rl = SimpleNamespace(algorithm=Options())
    self.metrics = SimpleNamespace(policy_update=0)
    self.obs = TensorDict(
      {
        "history": torch.randn(4, 225) * 0.01,
        "geometry": torch.rand(4, 768),
        "critic": torch.rand(4, 446),
      },
      batch_size=[4],
    )
    self.steps = 0

  def step(self, action):
    self.steps += 1
    self.obs = self.obs.clone()
    self.obs["history"][:, :12] += action * 0.001
    self.obs["geometry"] += 0.002
    reward = -action.square().mean(-1) + torch.arange(4) * 0.1
    return self.obs, reward, torch.zeros(4, dtype=torch.bool), {}


def test_teacher_entrypoint_updates_and_saves_fresh_learning_state(tmp_path):
  torch.manual_seed(47)
  threads = torch.get_num_threads()
  torch.set_num_threads(1)
  try:
    him, actor, cnn, gru = make_modules()
    teacher = Teacher(him, actor)
    initial_teacher = run.tensor_digest(teacher.state_dict())
    initial_him = run.tensor_digest(teacher.him.state_dict())
    env = SyntheticEnv()
    result = run.train_teacher(env, teacher, cnn, gru, 1, tmp_path)
    assert env.steps == 24
    assert result["completed_updates"] == 1 and result["optimizer_calls"] == 2
    assert run.tensor_digest(teacher.state_dict()) != initial_teacher
    assert run.tensor_digest(teacher.him.state_dict()) == initial_him
    _, _, _, saved = run.load_bundle(tmp_path / "teacher.pt")
    assert saved["learning_state"]["optimizer_calls"] == 2
    assert saved["qualification"] == "not_evaluated"
    metrics = json.loads((tmp_path / "updates.jsonl").read_text())
    assert metrics["him_next_velocity_rows"] == 96
    assert len(metrics["him_next_velocity_mae_xyz"]) == 3
  finally:
    torch.set_num_threads(threads)


def _fake_main(monkeypatch, tmp_path, train, close):
  """只替换入口边界，不创建环境、渲染器或优化器。"""
  teacher = SimpleNamespace(state_dict=lambda: {})
  env = SimpleNamespace(
    probe=SimpleNamespace(fault={"stage": "synthetic_native_failure"}),
    close=close,
    camera_receipt={},
    render_receipt={},
  )
  monkeypatch.setattr(run, "initial_models", lambda *_: (teacher, None, None))
  monkeypatch.setattr(run, "NativeEnv", lambda *_: env)
  monkeypatch.setattr(run, "train_teacher", train)
  monkeypatch.setattr(run.fcntl, "flock", lambda *_: None)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "run.py",
      "--mode",
      "teacher",
      "--execute",
      "--rounds",
      "1",
      "--device",
      "cpu",
      "--out",
      str(tmp_path / "candidate"),
    ],
  )
  return tmp_path / "candidate"


@pytest.mark.parametrize("failed_record_write", [False, True])
def test_original_exception_survives_diagnostic_and_close_errors(
  tmp_path, monkeypatch, failed_record_write
):
  original = RuntimeError("original native failure")
  diagnostic = OSError("fault snapshot write failed")
  cleanup = RuntimeError("environment close failed")
  recording = OSError("FAILED.json write failed")

  def train(*_, **kwargs):
    raise original

  def write_fault(*_):
    raise diagnostic

  def close():
    raise cleanup

  output = _fake_main(monkeypatch, tmp_path, train, close)
  monkeypatch.setitem(sys.modules, "probe", SimpleNamespace(write_fault=write_fault))
  if failed_record_write:
    actual_write = Path.write_text

    def reject_failed_record(path, *args, **kwargs):
      if path.name == "FAILED.json":
        raise recording
      return actual_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", reject_failed_record)

  with pytest.raises(RuntimeError) as raised:
    run.main()
  assert raised.value is original
  expected = {
    "error": repr(original),
    "diagnostic_error": repr(diagnostic),
    "close_error": repr(cleanup),
  }
  if failed_record_write:
    expected["failure_write_error"] = repr(recording)
    assert not (output / "FAILED.json").exists()
  else:
    assert json.loads((output / "FAILED.json").read_text()) == expected
  assert original.__dict__["failure_details"] == expected


def test_close_failure_after_success_is_still_raised(tmp_path, monkeypatch):
  cleanup = RuntimeError("environment close failed")

  def close():
    raise cleanup

  output = _fake_main(monkeypatch, tmp_path, lambda *_, **kwargs: {}, close)
  with pytest.raises(RuntimeError) as raised:
    run.main()
  assert raised.value is cleanup
  assert (output / "RESULT.json").is_file()
  assert json.loads((output / "FAILED.json").read_text()) == {"error": repr(cleanup)}
  assert cleanup.__dict__["failure_details"] == {"error": repr(cleanup)}
