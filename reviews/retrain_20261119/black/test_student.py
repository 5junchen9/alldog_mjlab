"""CPU验证共享权重、视觉梯度和连续帧/episode语义。"""

import pytest
import torch
from torch.nn import functional as F

from black.model import Teacher, make_modules
from black.student import SharedStudent, distillation_loss


@pytest.fixture
def example():
  old_threads = torch.get_num_threads()
  torch.set_num_threads(1)
  torch.manual_seed(19)
  him, actor, cnn, gru = make_modules()
  teacher = Teacher(him, actor)
  student = SharedStudent(teacher, cnn, gru)
  history = torch.randn(2, 5, 45)
  depth = torch.full((2, 48, 64, 1), 1.2)
  geometry = torch.randn(2, 768)
  yield teacher, student, history, depth, geometry
  torch.set_num_threads(old_threads)


def metadata(
  tick,
  *,
  episode=(10, 20),
  capture_episode=None,
  frame=(0, 1),
  capture_tick=(0, 0),
  mask=(True, True),
):
  return dict(
    tick=tick,
    episode=torch.tensor(episode),
    capture_episode=torch.tensor(
      episode if capture_episode is None else capture_episode
    ),
    frame=torch.tensor(frame),
    capture_tick=torch.tensor(capture_tick),
    mask=torch.tensor(mask),
  )


def test_shared_modules_frozen_even_when_student_enters_train(example):
  teacher, student, _, _, _ = example
  student.train()
  assert student.actor is teacher.actor and student.him is teacher.him
  assert not teacher.training and not student.him.training
  assert all(not parameter.requires_grad for parameter in teacher.parameters())
  selected = {id(parameter) for parameter in student.student_parameters()}
  assert selected == {
    id(parameter)
    for module in (student.depth_encoder, student.gru)
    for parameter in module.parameters()
  }
  assert not selected.intersection(id(parameter) for parameter in teacher.parameters())


def test_targets_are_teacher_means_and_detached_features(example):
  teacher, student, history, depth, geometry = example
  geometry.requires_grad_(True)
  _, visual, _ = student(history, depth, metadata(0))
  rng_before = torch.random.get_rng_state().clone()
  loss, metrics = distillation_loss(history, geometry, visual, student, teacher, 1.0)
  assert torch.equal(rng_before, torch.random.get_rng_state())
  with torch.no_grad():
    target = teacher.features(history, geometry)
    expected_action = F.huber_loss(
      student.act_with_visual(history, visual), teacher.actor(target)
    )
    expected_feature = F.huber_loss(visual, target[:, -32:])
  assert torch.equal(loss.detach(), expected_action + expected_feature)
  assert isinstance(metrics["action_huber"], torch.Tensor)
  assert isinstance(metrics["feature_huber"], torch.Tensor)
  assert torch.equal(metrics["action_huber"], expected_action)
  assert torch.equal(metrics["feature_huber"], expected_feature)
  loss.backward()
  assert geometry.grad is None
  assert all(parameter.grad is None for parameter in teacher.parameters())


def test_action_gradient_crosses_frozen_actor_then_reaches_gru(example):
  teacher, student, history, depth, geometry = example
  teacher_before = {name: value.clone() for name, value in teacher.state_dict().items()}
  visual_before = {
    name: value.clone() for name, value in student.depth_encoder.state_dict().items()
  }
  optimizer = torch.optim.SGD(student.student_parameters(), lr=0.05)

  _, visual, _ = student(history, depth, metadata(0))
  # feature_weight=0 isolates the gradient path through the frozen actor.
  loss, _ = distillation_loss(history, geometry, visual, student, teacher, 0.0)
  loss.backward()
  assert (
    sum(
      float(parameter.grad.abs().sum())
      for parameter in student.depth_encoder.parameters()
    )
    > 0
  )
  assert float(student.gru.projection.weight.grad.abs().sum()) > 0
  assert all(
    parameter.grad is None or not bool(parameter.grad.any())
    for parameter in student.gru.gru.parameters()
  )
  optimizer.step()
  optimizer.zero_grad(set_to_none=True)

  _, visual, _ = student(history, depth, metadata(0))
  loss, _ = distillation_loss(history, geometry, visual, student, teacher, 0.0)
  loss.backward()
  assert (
    sum(float(parameter.grad.abs().sum()) for parameter in student.gru.gru.parameters())
    > 0
  )
  optimizer.step()
  assert all(
    torch.equal(value, teacher.state_dict()[name])
    for name, value in teacher_before.items()
  )
  assert any(
    not torch.equal(value, student.depth_encoder.state_dict()[name])
    for name, value in visual_before.items()
  )
  assert all(parameter.grad is None for parameter in teacher.parameters())


def test_held_frame_does_not_advance_gru_and_reset_starts_new_episode(example):
  _, student, history, depth, _ = example
  rows = [
    metadata(0),
    metadata(1),
    metadata(2, episode=(11, 20), frame=(2, 1), capture_tick=(2, 0)),
    metadata(3, episode=(11, 20), frame=(2, 1), capture_tick=(2, 0)),
  ]
  with torch.no_grad():
    action0, visual0, state0 = student(history, depth, rows[0])
    action1, visual1, state1 = student(history, depth, rows[1], state0)
    assert torch.equal(state0.hidden, state1.hidden)
    assert torch.equal(visual0, visual1) and torch.equal(action0, action1)
    _, _, state2 = student(history, depth, rows[2], state1)
    _, _, fresh = student(history, depth, rows[2])
    # 一行fresh与两行fresh的矩阵乘法允许浮点尾差。
    torch.testing.assert_close(state2.hidden[0], fresh.hidden[0], rtol=1e-6, atol=1e-7)
    assert torch.equal(state2.hidden[1], state1.hidden[1])
    actions, visual, final = student.forward_sequence(
      history.unsqueeze(0).repeat(4, 1, 1, 1),
      depth.unsqueeze(0).repeat(4, 1, 1, 1, 1),
      rows,
    )
    assert actions.shape == (4, 2, 12) and visual.shape == (4, 2, 32)
    assert final.episode.tolist() == [11, 20]
    assert torch.equal(final.hidden, state2.hidden)


def test_real_frame_age_and_old_episode_mask_the_cnn_input(example):
  _, student, history, depth, _ = example
  observed = []
  handle = student.depth_encoder.register_forward_pre_hook(
    lambda _module, args: observed.append(args[0].detach().clone())
  )
  student(history, depth, metadata(7))  # age=.14s exceeds the original .12s.
  student(history, depth, metadata(0, capture_episode=(9, 19)))
  handle.remove()
  assert len(observed) == 2 and all(not bool(value.any()) for value in observed)


def test_future_frame_and_noncontiguous_control_are_rejected(example):
  _, student, history, depth, _ = example
  with pytest.raises(ValueError, match="future"):
    student(history, depth, metadata(0, capture_tick=(1, 1)))
  _, _, state = student(history, depth, metadata(0))
  with pytest.raises(ValueError, match="discontinuity"):
    student(history, depth, metadata(2), state)


def test_frame_age_boundary_survives_a_long_control_clock(example):
  _, student, history, depth, _ = example
  observed = []
  handle = student.depth_encoder.register_forward_pre_hook(
    lambda _module, args: observed.append(args[0].detach().clone())
  )
  student(history, depth, metadata(100000006, capture_tick=(100000000, 100000000)))
  student(history, depth, metadata(100000007, capture_tick=(100000000, 100000000)))
  handle.remove()
  assert bool(observed[0][:, 1].all())  # 第6步仍在0.12s有效窗内。
  assert not bool(observed[1].any())
