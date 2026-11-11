"""HIM学生的数值、梯度隔离和终止标签检查。"""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.black_him_student import (
  HIMStudent,
  HIMStudentPolicy,
  sinkhorn,
  update_student,
)
from mjlab.scripts.train_black_him_student import collect_student_batch


def test_huber_limits_outlier_penalty_and_keeps_labels_detached():
  student = HIMStudent()
  with torch.no_grad():
    encoder_last = cast(torch.nn.Linear, student.encoder[-1])
    actor_last = cast(torch.nn.Linear, student.actor[-1])
    assert encoder_last.bias is not None and actor_last.bias is not None
    encoder_last.weight[:3].zero_()
    encoder_last.bias[:3].zero_()
    actor_last.weight.zero_()
    actor_last.bias.zero_()
  history = torch.randn(8, 5, 45)
  teacher_action = torch.full((8, 12), 4.0, requires_grad=True)
  velocity = torch.full((8, 3), 4.0, requires_grad=True)
  actor_optimizer = torch.optim.Adam(student.actor.parameters(), lr=1e-3)
  estimator_optimizer = torch.optim.Adam(student.estimator_parameters(), lr=1e-3)
  losses = update_student(
    student,
    actor_optimizer,
    estimator_optimizer,
    history,
    teacher_action,
    torch.randn(8, 45),
    velocity,
  )
  # 误差4对应Huber=4-0.5=3.5；MSE会得到16。
  assert losses["action_huber"] == pytest.approx(3.5)
  assert losses["velocity_huber"] == pytest.approx(3.5)
  assert teacher_action.grad is None
  assert velocity.grad is None
  assert all(torch.isfinite(p).all() for p in student.parameters())


def test_actor_and_estimator_gradients_are_separate():
  student = HIMStudent()
  history = torch.randn(8, 5, 45)
  student(history).square().mean().backward()
  assert any(p.grad is not None for p in student.actor.parameters())
  assert all(p.grad is None for p in student.estimator_parameters())
  student.zero_grad(set_to_none=True)
  velocity, contrastive = student.estimator_losses(
    history, torch.randn(8, 45), torch.randn(8, 3)
  )
  (velocity + contrastive).backward()
  assert all(p.grad is None for p in student.actor.parameters())
  for module in (student.encoder, student.target_encoder, student.prototypes):
    assert any(
      p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
    )


def test_policy_export_needs_only_history(tmp_path):
  student = HIMStudent().eval()
  history = torch.randn(3, 5, 45)
  policy = HIMStudentPolicy(student).eval()
  path = tmp_path / "student_policy.pt"
  torch.jit.script(policy).save(str(path))
  exported = torch.jit.load(str(path))
  with torch.no_grad():
    expected = student(history)
    actual = exported(history)
  assert actual.shape == (3, 12)
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  assert all(
    "target_encoder" not in key and "prototypes" not in key
    for key in exported.state_dict()
  )


def test_sinkhorn_assignments_are_finite_probabilities():
  scores = torch.randn(64, 32)
  assignment = sinkhorn(scores)
  assert torch.isfinite(assignment).all()
  assert (assignment >= 0).all()
  torch.testing.assert_close(assignment.sum(dim=1), torch.ones(64))
  assert not assignment.requires_grad


def test_collector_uses_terminal_successor_and_same_state_teacher():
  step_count = 0

  def observation(value):
    return TensorDict(
      {
        "history": torch.full((1, 5, 45), float(value)),
        "actor": torch.full((1, 72), float(value)),
      },
      batch_size=[1],
    )

  def step(actions):
    nonlocal step_count
    step_count += 1
    return observation(step_count), torch.zeros(1), torch.ones(1), {}

  def reset(env_ids):
    return observation(100 + step_count).to_dict(), {}

  def get_observations():
    return observation(100 + step_count)

  env = SimpleNamespace(
    clip_actions=None,
    num_envs=1,
    step=step,
    get_observations=get_observations,
    unwrapped=SimpleNamespace(reset=reset),
  )

  def teacher(obs):
    return obs["actor"][:, :12]

  def student(history):
    return torch.zeros(history.shape[0], 12)

  _, batch = collect_student_batch(
    cast(RslRlVecEnvWrapper, env),
    cast(torch.nn.Module, teacher),
    cast(HIMStudent, student),
    observation(0),
  )
  # 下一帧必须是终止前的1，不能变成reset后的101。
  torch.testing.assert_close(batch["next_frame"][0], torch.ones(45))
  torch.testing.assert_close(batch["next_velocity"][0], torch.ones(3))
  # 第二条标签查询的是学生reset后真正到达的101状态。
  torch.testing.assert_close(batch["teacher_action"][0], torch.zeros(12))
  torch.testing.assert_close(batch["teacher_action"][1], torch.full((12,), 101.0))
