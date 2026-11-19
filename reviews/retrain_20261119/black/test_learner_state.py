"""CPU检查真实随机序列与恢复身份；CUDA测试只替换接口，不初始化GPU。"""

import copy
import io
import random

import numpy as np
import pytest
import torch

from black.learner_state import (
  capture_rng,
  parameter_signature,
  restore_rng,
  validate_resume,
)


@pytest.fixture(autouse=True)
def preserve_global_rng():
  original = capture_rng("cpu")
  yield
  restore_rng(original, "cpu")


def test_cpu_rng_round_trip_is_weights_only_and_repeats_next_draws():
  random.seed(19)
  np.random.seed(19)
  torch.manual_seed(19)
  # 将Python和NumPy高斯缓存也置为非空，避免只验证整数种子。
  random.gauss(0, 1)
  np.random.normal()
  saved = capture_rng("cpu")
  stream = io.BytesIO()
  torch.save(saved, stream)
  stream.seek(0)
  loaded = torch.load(stream, weights_only=True)

  def draws():
    return random.gauss(0, 1), np.random.normal(size=8), torch.rand(8)

  expected = draws()
  draws()
  restore_rng(loaded, "cpu")
  actual = draws()
  assert actual[0] == expected[0]
  assert np.array_equal(actual[1], expected[1])
  assert torch.equal(actual[2], expected[2])
  assert loaded["cuda"] is None


def test_cuda_rng_uses_requested_device_without_gpu(monkeypatch):
  calls = []
  state = torch.tensor([7, 11, 19], dtype=torch.uint8)
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

  def get_state(device):
    calls.append(("get", str(device)))
    return state

  def set_state(value, device):
    calls.append(("set", str(device)))
    assert torch.equal(value, state)

  monkeypatch.setattr(torch.cuda, "get_rng_state", get_state)
  monkeypatch.setattr(torch.cuda, "set_rng_state", set_state)
  saved = capture_rng("cuda:3")
  restore_rng(saved, "cuda:3")
  assert calls == [("get", "cuda:3"), ("set", "cuda:3")]
  assert saved["cuda"].data_ptr() != state.data_ptr()
  with pytest.raises(ValueError, match="explicit device index"):
    capture_rng("cuda")
  with pytest.raises(ValueError, match="CUDA target"):
    restore_rng(saved, "cpu")


def test_unavailable_cuda_has_no_state_or_device_calls(monkeypatch):
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

  def unexpected(*_):
    raise AssertionError("CUDA RNG must not be called when unavailable")

  monkeypatch.setattr(torch.cuda, "get_rng_state", unexpected)
  monkeypatch.setattr(torch.cuda, "set_rng_state", unexpected)
  saved = capture_rng("cuda:2")
  assert saved["cuda"] is None
  with pytest.raises(ValueError, match="CUDA RNG is missing"):
    restore_rng(saved, "cuda:2")


def test_cpu_rng_packet_cannot_claim_complete_cuda_restore(monkeypatch):
  saved = capture_rng("cpu")
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

  def unexpected(*_):
    raise AssertionError("missing CUDA state must be rejected before device restore")

  monkeypatch.setattr(torch.cuda, "set_rng_state", unexpected)
  with pytest.raises(ValueError, match="CUDA RNG is missing"):
    restore_rng(saved, "cuda:0")


def test_resume_requires_phase_schema_recipe_and_integer_counters():
  recipe = {
    "ppo_options": {"learning_rate": 1e-4},
    "num_envs": 32,
    "controls_per_round": 24,
  }
  source = {"sha256": "test-source", "runtime": {"torch": "test"}}
  packet: dict = {
    "source_identity": source,
    "phase": "teacher",
    "completed_updates": 1,
    "learning_state": {
      "schema": "black_retrain_learner_v1",
      "recipe": recipe,
      "optimizer_calls": 20,
    },
  }
  assert (
    validate_resume(packet, "teacher", copy.deepcopy(recipe), current_source=source)
    is packet["learning_state"]
  )
  bad = copy.deepcopy(packet)
  bad["phase"] = "student_distillation"
  with pytest.raises(ValueError, match="phase"):
    validate_resume(bad, "teacher", recipe, current_source=source)
  bad = copy.deepcopy(packet)
  bad["learning_state"]["schema"] = "legacy"
  with pytest.raises(ValueError, match="schema"):
    validate_resume(bad, "teacher", recipe, current_source=source)
  bad = copy.deepcopy(packet)
  bad["learning_state"]["recipe"]["ppo_options"]["learning_rate"] = 2e-4
  with pytest.raises(ValueError, match="recipe"):
    validate_resume(bad, "teacher", recipe, current_source=source)
  bad = copy.deepcopy(packet)
  bad["learning_state"]["recipe"]["controls_per_round"] = 24.0
  with pytest.raises(ValueError, match="recipe"):
    validate_resume(bad, "teacher", recipe, current_source=source)
  for value in (0, -1, True, 1.0, None):
    bad = copy.deepcopy(packet)
    bad["completed_updates"] = value
    with pytest.raises(ValueError, match="completed_updates"):
      validate_resume(bad, "teacher", recipe, current_source=source)
  for value in (-1, True, 1.0, None):
    bad = copy.deepcopy(packet)
    bad["learning_state"]["optimizer_calls"] = value
    with pytest.raises(ValueError, match="optimizer_calls"):
      validate_resume(bad, "teacher", recipe, current_source=source)
  bad = copy.deepcopy(packet)
  bad["learning_state"]["optimizer_calls"] = 0
  assert (
    validate_resume(bad, "teacher", recipe, current_source=source)["optimizer_calls"]
    == 0
  )


def test_parameter_signature_preserves_binding_order_shapes_and_dtypes():
  matrix = torch.nn.Parameter(torch.zeros(3, 2))
  vector = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
  named = [("cnn.weight", matrix), ("gru.bias", vector)]
  assert parameter_signature(iter(named)) == [
    ["cnn.weight", [3, 2], "torch.float32"],
    ["gru.bias", [2], "torch.float64"],
  ]
  assert parameter_signature(reversed(named)) != parameter_signature(named)
