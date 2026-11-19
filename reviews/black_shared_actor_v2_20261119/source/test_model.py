"""全随机HIM及共享头的梯度边界；不读取旧档或创建仿真。"""

import torch
from checkpoint import initial_models, tensor_digest
from model import Teacher, make_modules
from torch.nn import functional as F


def test_scratch_never_reads_weights_and_repeats_given_seed(monkeypatch):
  def forbidden(*args, **kwargs):
    raise AssertionError("scratch must not load a checkpoint")

  monkeypatch.setattr(torch, "load", forbidden)
  torch.manual_seed(742)
  first = initial_models()
  torch.manual_seed(742)
  second = initial_models()
  assert all(
    tensor_digest(a.state_dict()) == tensor_digest(b.state_dict())
    for a, b in zip(first, second, strict=True)
  )
  torch.manual_seed(743)
  third = initial_models()
  assert all(
    tensor_digest(a.state_dict()) != tensor_digest(b.state_dict())
    for a, b in zip(first, third, strict=True)
  )


def test_him_preserves_next_velocity_and_unit_latent_semantics():
  him, _, _, _ = make_modules()
  history = torch.randn(4, 5, 45)
  encoded = him.encoder(history.flatten(start_dim=1))
  actual = him(history)
  assert torch.equal(actual[:, :3], encoded[:, :3])
  assert torch.equal(actual[:, 3:], F.normalize(encoded[:, 3:], dim=-1))
  assert not actual.requires_grad
  assert hasattr(him, "target_encoder") and hasattr(him, "prototypes")


def test_teacher_gradient_updates_head_and_terrain_but_not_him():
  him, actor, _, _ = make_modules()
  teacher = Teacher(him, actor).train()
  history = torch.randn(4, 5, 45)
  before = {name: value.clone() for name, value in teacher.state_dict().items()}
  optimizer = torch.optim.Adam(
    [p for p in teacher.parameters() if p.requires_grad], lr=1e-4
  )
  teacher(history, torch.randn(4, 768)).square().mean().backward()
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


def test_frozen_shared_actor_still_backpropagates_to_student_visual():
  him, actor, _, _ = make_modules()
  teacher = Teacher(him, actor).freeze()
  history = torch.randn(4, 5, 45)
  visual = torch.randn(4, 32, requires_grad=True)
  features = torch.cat((history[:, -1], him(history), visual), dim=-1)
  teacher.actor(features).square().mean().backward()
  assert visual.grad is not None and bool(visual.grad.abs().sum() > 0)
  assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
