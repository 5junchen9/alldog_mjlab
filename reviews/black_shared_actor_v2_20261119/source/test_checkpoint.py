"""写入中断与重复目标均不能破坏已发布的权重包。"""

import checkpoint
import pytest
import torch
from model import Teacher, make_modules


@pytest.fixture
def modules():
  previous_threads = torch.get_num_threads()
  torch.set_num_threads(1)
  him, actor, cnn, gru = make_modules()
  yield Teacher(him, actor), cnn, gru
  torch.set_num_threads(previous_threads)


def test_interrupted_write_does_not_publish_or_damage_previous_bundle(
  modules,
  tmp_path,
  monkeypatch,
):
  previous = tmp_path / "student_1.pt"
  checkpoint.save_bundle(previous, *modules, phase="student_distillation", updates=1)
  saved_bytes = previous.read_bytes()
  target = tmp_path / "student_2.pt"

  def interrupted_save(_packet, stream):
    stream.write(b"incomplete checkpoint bytes")
    stream.flush()
    assert not target.exists()
    raise OSError("injected interrupted write")

  monkeypatch.setattr(torch, "save", interrupted_save)
  with pytest.raises(OSError, match="injected interrupted write"):
    checkpoint.save_bundle(target, *modules, phase="student_distillation", updates=2)
  assert not target.exists()
  assert previous.read_bytes() == saved_bytes
  assert torch.load(previous, weights_only=True)["completed_updates"] == 1
  assert set(tmp_path.iterdir()) == {previous}


def test_existing_target_is_not_overwritten(modules, tmp_path):
  target = tmp_path / "student.pt"
  checkpoint.save_bundle(target, *modules, phase="student_distillation", updates=1)
  saved_bytes = target.read_bytes()
  # 第二份包确实含不同内容，拒绝覆盖后仍应读取第一份。
  with torch.no_grad():
    next(modules[1].parameters()).add_(1)
  with pytest.raises(FileExistsError):
    checkpoint.save_bundle(target, *modules, phase="student_distillation", updates=2)
  assert target.read_bytes() == saved_bytes
  _, restored_cnn, _, packet = checkpoint.load_bundle(target)
  assert packet["completed_updates"] == 1
  assert not torch.equal(next(restored_cnn.parameters()), next(modules[1].parameters()))
  assert set(tmp_path.iterdir()) == {target}


def test_v2_provenance_roundtrip_and_legacy_contract_rejection(modules, tmp_path):
  path = tmp_path / "new.pt"
  checkpoint.save_bundle(path, *modules, phase="probe", updates=0)
  _, _, _, packet = checkpoint.load_bundle(path)
  assert packet["contract"]["schema"] == "black_shared_actor_v2"
  assert packet["initialization"]["mode"] == "scratch"
  assert set(packet["initialization"]["modules"]) == {
    "HIM",
    "actor",
    "terrain_encoder",
    "CNN",
    "GRU",
  }
  assert "initial_source_sha256" not in packet
  packet["contract"] = dict(packet["contract"], schema="black_shared_actor_v1")
  torch.save(packet, path)
  with pytest.raises(ValueError, match="输入约定"):
    checkpoint.load_bundle(path)
