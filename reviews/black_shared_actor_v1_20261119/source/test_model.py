"""真实r800权重的接线与梯度检查；没有环境或GPU训练。"""

from pathlib import Path

import pytest
import torch
from model import Teacher, make_initial_modules
from torch.nn import functional as F

SAMPLE = (
  Path(__file__).resolve().parents[1]
  / "black_reliability_20261119/evidence/WINDOWS_R800_SAMPLE.pt"
)


@pytest.fixture(scope="module")
def sample():
  return torch.load(SAMPLE, map_location="cpu", weights_only=True)


def test_r800_weights_and_him_meaning_are_preserved(sample):
  him, actor, depth_encoder, gru = make_initial_modules(sample["learner"])
  history = sample["rows"][0]["history"][:4]
  encoded = him.encoder(history.flatten(start_dim=1))
  actual = him(history)
  assert torch.equal(actual[:, :3], encoded[:, :3])
  assert torch.equal(actual[:, 3:], F.normalize(encoded[:, 3:], dim=-1))
  assert not actual.requires_grad
  for name, value in actor.state_dict().items():
    assert torch.equal(value, sample["learner"]["head"][name])
  for name, value in him.encoder.state_dict().items():
    assert torch.equal(value, sample["learner"]["frontend"]["encoder." + name])
  for name, value in depth_encoder.state_dict().items():
    assert torch.equal(value, sample["learner"]["frontend"]["depth_encoder." + name])
  for name, value in gru.state_dict().items():
    assert torch.equal(value, sample["learner"]["gru"][name])


def test_teacher_gradient_updates_head_and_terrain_but_not_him(sample):
  torch.manual_seed(742)
  him, actor, _, _ = make_initial_modules(sample["learner"])
  teacher = Teacher(him, actor).train()
  history = sample["rows"][0]["history"][:4]
  geometry = torch.randn(4, 768)
  before = {name: value.clone() for name, value in teacher.state_dict().items()}
  optimizer = torch.optim.Adam(
    [p for p in teacher.parameters() if p.requires_grad], lr=1e-4
  )
  teacher(history, geometry).square().mean().backward()
  assert not teacher.him.training
  assert all(p.grad is None for p in teacher.him.parameters())
  assert all(p.grad is not None for p in teacher.actor.parameters())
  assert all(p.grad is not None for p in teacher.terrain_encoder.parameters())
  optimizer.step()
  after = teacher.state_dict()
  for prefix in ("actor.", "terrain_encoder."):
    assert any(
      not torch.equal(before[name], after[name])
      for name in before
      if name.startswith(prefix)
    )
  assert all(
    torch.equal(before[name], after[name]) for name in before if name.startswith("him.")
  )


def test_frozen_shared_actor_still_backpropagates_to_student_visual(sample):
  him, actor, _, _ = make_initial_modules(sample["learner"])
  teacher = Teacher(him, actor).freeze()
  history = sample["rows"][0]["history"][:4]
  visual = torch.randn(4, 32, requires_grad=True)
  features = torch.cat((history[:, -1], him(history), visual), dim=-1)
  teacher.actor(features).square().mean().backward()
  assert visual.grad is not None and bool(visual.grad.abs().sum() > 0)
  assert all(p.grad is None for p in teacher.parameters())
  assert all(not p.requires_grad for p in teacher.parameters())
