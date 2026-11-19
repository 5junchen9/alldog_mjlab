"""写入中断与重复目标均不能破坏已发布的权重包。"""

import pytest
import torch

import black.checkpoint as checkpoint
from black.model import Teacher, make_modules


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
  assert packet["contract"]["schema"] == "black_retrain_shared_actor_v2"
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


@pytest.mark.parametrize("relative", ["src/task/rewards.py", "src/robot/collision.STL"])
def test_changed_runtime_code_or_asset_blocks_model_loading(
  modules, tmp_path, monkeypatch, relative
):
  import black.eval_collect as collector
  from black.test_source_identity import select_root

  source = tmp_path / "source"
  file = source / relative
  file.parent.mkdir(parents=True)
  file.write_text("original")
  select_root(monkeypatch, source)
  path = tmp_path / "teacher.pt"
  checkpoint.save_bundle(path, *modules, phase="teacher", updates=1)
  before = collector.source_files()
  file.write_text("changed physical or reward semantics")
  assert before != collector.source_files()
  with pytest.raises(ValueError, match="源码或资产"):
    checkpoint.load_bundle(path)


def test_runtime_change_allows_weight_transfer_but_rejects_optimizer_resume(
  modules, tmp_path, monkeypatch
):
  import black.eval_collect as collector
  from black.learner_state import SCHEMA, validate_resume

  recipe = {"controls": 24}
  path = tmp_path / "teacher.pt"
  checkpoint.save_bundle(
    path,
    *modules,
    phase="teacher",
    updates=1,
    learning_state=dict(schema=SCHEMA, recipe=recipe, optimizer_calls=20),
  )
  identity = checkpoint.source_identity()
  runtime = dict(identity["runtime"], torch_cuda_build="different-CUDA-build")
  monkeypatch.setattr(collector, "runtime_identity", lambda: runtime)
  current = checkpoint.source_identity()
  assert current["sha256"] == identity["sha256"]
  assert current["runtime"] != identity["runtime"]
  teacher, _, _, packet = checkpoint.load_bundle(path)
  assert checkpoint.teacher_identity(teacher) == packet["teacher_id"]
  with pytest.raises(ValueError, match="source/runtime"):
    validate_resume(packet, "teacher", recipe, current_source=current)


def test_source_relocation_loads_same_bundle_and_tests_docs_do_not_change_identity(
  modules, tmp_path, monkeypatch
):
  import shutil

  import black.eval_collect as collector
  from black.test_source_identity import select_root

  first, second = tmp_path / "first", tmp_path / "relocated"
  for relative in (
    "black/model.py",
    "src/robot/collision.STL",
    "support/black/parameters.py",
  ):
    path = first / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(relative)
  select_root(monkeypatch, first)
  path = tmp_path / "teacher.pt"
  checkpoint.save_bundle(path, *modules, phase="teacher", updates=1)
  before = checkpoint.source_identity()
  (first / "black/test_model.py").write_text("new regression test")
  (first / "docs").mkdir()
  (first / "docs/guide.md").write_text("edited documentation")
  (first / "src/tests").mkdir()
  (first / "src/tests/test_asset.py").write_text("asset fixture")
  assert checkpoint.source_identity() == before
  shutil.copytree(first, second)
  select_root(monkeypatch, second)
  assert checkpoint.source_identity() == before
  _, _, _, packet = checkpoint.load_bundle(path)
  assert packet["source_identity"] == before
  assert not any(
    "test_" in name or "/tests/" in name for name in collector.source_files()
  )


def test_checkpoint_without_code_identity_cannot_load(modules, tmp_path):
  path = tmp_path / "teacher.pt"
  checkpoint.save_bundle(path, *modules, phase="teacher", updates=1)
  packet = torch.load(path, weights_only=True)
  del packet["source_identity"]
  torch.save(packet, path)
  with pytest.raises(ValueError, match="源码或资产"):
    checkpoint.load_bundle(path)
