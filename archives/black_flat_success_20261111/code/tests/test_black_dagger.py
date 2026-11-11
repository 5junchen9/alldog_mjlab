"""检查DAgger跨轮次聚合、样本对应和续训状态。"""

from types import SimpleNamespace
from typing import cast

import torch
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.black_dagger import DAggerDataset
from mjlab.rl.black_him_student import HIMStudent, update_student
from mjlab.scripts.train_black_him_student import collect_student_batch


def labelled_batch(start, count):
  values = torch.arange(start, start + count, dtype=torch.float32)
  history = values[:, None, None].expand(-1, 5, 45).clone()
  actions = (values * 10)[:, None].expand(-1, 12).clone()
  return history, actions


def test_dagger_keeps_earlier_rounds_and_detaches_source():
  dataset = DAggerDataset(capacity=20)
  history, action = labelled_batch(0, 4)
  dataset.add(history, action)
  history.fill_(-1)
  dataset.add(*labelled_batch(4, 4))
  assert dataset.total_seen == 8
  assert len(dataset) == 8
  torch.testing.assert_close(dataset.history[:, 0, 0], torch.arange(8).float())
  assert dataset.history.device.type == "cpu"
  sample = dataset.sample(64, "cpu")
  torch.testing.assert_close(
    sample["teacher_action"][:, 0], sample["history"][:, 0, 0] * 10
  )


def test_capacity_preserves_label_pairing_and_uniform_history_subset():
  torch.manual_seed(42)
  dataset = DAggerDataset(capacity=100)
  dataset.add(*labelled_batch(0, 100))
  dataset.add(*labelled_batch(100, 100))
  assert len(dataset) == 100 and dataset.total_seen == 200
  # 上限触发后，保留样本来自前后两轮，而非只保留最后一轮。
  values = dataset.history[:, 0, 0]
  assert (values < 100).any() and (values >= 100).any()
  torch.testing.assert_close(dataset.teacher_action[:, 0], values * 10)
  assert dataset.priorities.numel() == 100


def test_checkpoint_restores_dataset_and_continues_aggregation(tmp_path):
  dataset = DAggerDataset(capacity=8)
  dataset.add(*labelled_batch(0, 6))
  path = tmp_path / "dagger.pt"
  torch.save(dataset.state_dict(), path)
  restored = DAggerDataset()
  restored.load_state_dict(torch.load(path, weights_only=False))
  assert restored.capacity == 8 and restored.total_seen == 6
  torch.testing.assert_close(restored.history, dataset.history)
  torch.testing.assert_close(restored.priorities, dataset.priorities)
  torch.manual_seed(7)
  dataset.add(*labelled_batch(6, 6))
  torch.manual_seed(7)
  restored.add(*labelled_batch(6, 6))
  torch.testing.assert_close(restored.history, dataset.history)
  torch.testing.assert_close(restored.teacher_action, dataset.teacher_action)
  expected = dataset.sample(16, "cpu")
  actual = restored.sample(16, "cpu")
  torch.testing.assert_close(actual["history"], expected["history"])


def test_action_update_can_use_dagger_history_without_replacing_him_transition():
  student = HIMStudent()
  actor_optimizer = torch.optim.Adam(student.actor.parameters(), lr=1e-3)
  estimator_optimizer = torch.optim.Adam(student.estimator_parameters(), lr=1e-3)
  # 聚合动作批次和当前HIM转移批次大小不同，用以检测串用了历史输入。
  losses = update_student(
    student,
    actor_optimizer,
    estimator_optimizer,
    history=torch.randn(8, 5, 45),
    teacher_action=torch.randn(5, 12),
    next_frame=torch.randn(8, 45),
    next_velocity=torch.randn(8, 3),
    dagger_history=torch.randn(5, 5, 45),
  )
  assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_teacher_warmup_executes_teacher_and_student_round_executes_student():
  obs = TensorDict(
    {
      "history": torch.zeros(1, 5, 45),
      "actor": torch.zeros(1, 72),
    },
    batch_size=[1],
  )
  executed = []

  def step(actions):
    executed.append(actions.clone())
    return obs, torch.zeros(1), torch.zeros(1), {}

  def teacher(observations):
    return torch.full((1, 12), 2.0)

  def student(history):
    return torch.full((1, 12), -3.0)

  env = SimpleNamespace(clip_actions=None, step=step)
  collect_student_batch(
    cast(RslRlVecEnvWrapper, env),
    cast(torch.nn.Module, teacher),
    cast(HIMStudent, student),
    obs,
    use_teacher=True,
  )
  assert all(torch.equal(value, torch.full((1, 12), 2.0)) for value in executed)
  executed.clear()
  _, batch = collect_student_batch(
    cast(RslRlVecEnvWrapper, env),
    cast(torch.nn.Module, teacher),
    cast(HIMStudent, student),
    obs,
    use_teacher=False,
  )
  assert all(torch.equal(value, torch.full((1, 12), -3.0)) for value in executed)
  torch.testing.assert_close(batch["teacher_action"], torch.full((24, 12), 2.0))
