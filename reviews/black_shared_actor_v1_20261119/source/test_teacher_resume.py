"""相同合成轨迹的学习器续算；不声称保存或恢复了真实物理状态。"""

import copy

import pytest
import run
import torch
from learner_state import capture_rng, restore_rng
from model import Teacher, make_modules
from test_entrypoint import SyntheticEnv


@pytest.fixture
def modules():
  threads = torch.get_num_threads()
  torch.set_num_threads(1)
  torch.manual_seed(704)
  him, actor, cnn, gru = make_modules()
  yield Teacher(him, actor), cnn, gru
  torch.set_num_threads(threads)


def assert_same(left, right):
  if isinstance(left, torch.Tensor):
    torch.testing.assert_close(left, right, rtol=0, atol=0)
  elif isinstance(left, dict):
    assert left.keys() == right.keys()
    for key in left:
      assert_same(left[key], right[key])
  elif isinstance(left, (tuple, list)):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
      assert_same(a, b)
  else:
    assert left == right


def test_teacher_resume_matches_uninterrupted_learning_on_same_synthetic_states(
  modules, tmp_path
):
  env = SyntheticEnv()
  branches = [copy.deepcopy(modules), copy.deepcopy(modules)]
  full, split, resumed = [tmp_path / name for name in ("full", "split", "resumed")]
  for path in (full, split, resumed):
    path.mkdir()
  rng = capture_rng("cpu")
  teacher, cnn, gru = branches[0]
  run.train_teacher(copy.deepcopy(env), teacher, cnn, gru, 2, full, save_every=1)
  restore_rng(rng, "cpu")
  teacher, cnn, gru = branches[1]
  run.train_teacher(env, teacher, cnn, gru, 1, split, save_every=1)
  teacher, cnn, gru, packet = run.load_bundle(split / "teacher.pt")
  result = run.train_teacher(
    env, teacher, cnn, gru, 1, resumed, resume=packet, save_every=1
  )
  _, _, _, actual = run.load_bundle(resumed / "teacher.pt")
  _, _, _, expected = run.load_bundle(full / "teacher.pt")
  assert_same(actual, expected)
  assert result["starting_updates"] == result["added_updates"] == 1
  assert result["completed_updates"] == 2 and result["optimizer_calls"] == 4
  assert result["learner_resumed"] and not result["exact_physical_resume"]
  assert (resumed / "teacher_update_000002.pt").is_file()
  assert not (resumed / "teacher_update_000001.pt").exists()
  optimizer = actual["learning_state"]["ppo"]["optimizer_state_dict"]
  assert all(int(value["step"]) == 4 for value in optimizer["state"].values())
  normalizer = actual["learning_state"]["ppo"]["critic_state_dict"]
  assert normalizer["obs_normalizer.raw72.count"].item() == 192
  assert normalizer["obs_normalizer.terrain.count"].item() == 192


@pytest.mark.parametrize("changed", ["recipe", "signature", "inner_teacher", "legacy"])
def test_teacher_resume_rejects_incompatible_learning_state_before_collecting(
  modules, tmp_path, changed
):
  teacher, cnn, gru = modules
  run.train_teacher(SyntheticEnv(), teacher, cnn, gru, 1, tmp_path)
  teacher, cnn, gru, packet = run.load_bundle(tmp_path / "teacher.pt")
  saved = packet["learning_state"]
  if changed == "recipe":
    saved["recipe"]["num_envs"] += 1
  elif changed == "signature":
    saved["parameter_signature"].reverse()
  elif changed == "inner_teacher":
    saved["ppo"]["actor_state_dict"]["policy.terrain_encoder.0.weight"] += 1
  else:
    del saved["schema"]
  env = SyntheticEnv()
  with pytest.raises(ValueError):
    run.train_teacher(env, teacher, cnn, gru, 1, tmp_path, resume=packet)
  assert env.steps == 0


def test_teacher_failure_keeps_last_complete_periodic_bundle(
  modules, tmp_path, monkeypatch
):
  teacher, cnn, gru = modules
  env = SyntheticEnv()
  original = env.step

  def step(action):
    if env.steps == 24:
      raise RuntimeError("synthetic second round failure")
    return original(action)

  monkeypatch.setattr(env, "step", step)
  with pytest.raises(RuntimeError, match="second round failure"):
    run.train_teacher(env, teacher, cnn, gru, 2, tmp_path, save_every=1)
  _, _, _, saved = run.load_bundle(tmp_path / "teacher_update_000001.pt")
  assert saved["completed_updates"] == 1
  assert saved["learning_state"]["optimizer_calls"] == 2
  assert not (tmp_path / "teacher.pt").exists()
  assert not (tmp_path / "teacher_update_000002.pt").exists()
