"""CPU regressions for student-only selective PPO continuation."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.rl.black_continuation import (
  RecentDAgger,
  SelectiveTeacherPPO,
  masked_huber,
  teacher_selection,
)


def test_teacher_domains_and_exclusions():
  command = torch.tensor(
    [
      [0.2, 0.1, 0.1],
      [0.7, 0.0, 0.0],
      [1.2, 0.8, 0.6],
      [0.2, 0.0, 0.0],
      [0.2, 0.0, 0.0],
      [float("nan"), 0.0, 0.0],
      [0.7, 0.0, 0.0],
      [0.7, 0.0, 0.0],
      [0.7, 0.0, 0.0],
    ]
  )
  target = command.clone()
  target[6] = torch.tensor([float("nan"), 0.0, 0.0])
  target[7] = torch.tensor([float("inf"), 0.0, 0.0])
  target[8] = torch.tensor([1.2, 0.8, 0.6])
  healthy = torch.ones(len(command), dtype=torch.bool)
  healthy[3] = False
  age = torch.ones(len(command))
  age[4] = 0.99
  selected = teacher_selection(command, target, healthy, age)
  assert selected.tolist() == [2, 2, 0, 0, 0, 0, 0, 0, 0]
  assert not ((selected == 1) & (selected == 2)).any()


def test_masked_huber_weights_detached_labels_and_empty_mask():
  prediction = torch.tensor([[1.0], [2.0], [99.0]], requires_grad=True)
  labels = torch.zeros_like(prediction, requires_grad=True)
  loss = masked_huber(prediction, labels, torch.tensor([1, 2, 0]), (0.2, 0.1))
  assert loss.item() == pytest.approx(0.2 * 0.375 + 0.1 * 0.875)
  loss.backward()
  assert labels.grad is None
  torch.testing.assert_close(prediction.grad, torch.tensor([[0.1], [0.05], [0.0]]))
  prediction.grad = None
  empty = masked_huber(prediction, labels, torch.zeros(3), (0.2, 0.1))
  empty.backward()
  assert empty.item() == 0.0
  assert prediction.grad is not None
  assert torch.count_nonzero(prediction.grad) == 0


def test_recent_fifo_alignment_capacity_and_rng_resume():
  replay = RecentDAgger(capacity=5)

  def add(start, stop):
    ids = torch.arange(start, stop, dtype=torch.float32)
    replay.add(
      ids[:, None].expand(-1, 72),
      (ids * 10)[:, None].expand(-1, 12),
      (ids.long() % 2) + 1,
    )

  add(0, 3)
  add(3, 8)
  assert replay.size == 5 and replay.total_seen == 8
  assert set(replay.obs[:, 0].tolist()) == set(range(3, 8))
  add(8, 16)
  assert replay.size == 5 and replay.total_seen == 16
  assert set(replay.obs[:, 0].tolist()) == set(range(11, 16))
  replay.add(torch.zeros(2, 72), torch.zeros(2, 12), torch.zeros(2).long())
  assert replay.total_seen == 16
  restored = RecentDAgger(1)
  restored.load_state_dict(deepcopy(replay.state_dict()))
  for _ in range(3):
    sample = replay.sample(17, "cpu")
    resumed = restored.sample(17, "cpu")
    for a, b in zip(sample, resumed, strict=True):
      torch.testing.assert_close(a, b, rtol=0, atol=0)
    obs, action, teacher = sample
    torch.testing.assert_close(action[:, 0], 10 * obs[:, 0])
    torch.testing.assert_close(teacher, obs[:, 0].long() % 2 + 1)
    assert not obs.requires_grad and not action.requires_grad


def make_algorithm(cls, *, clipped=True, normalize=False):
  torch.manual_seed(19)
  obs = TensorDict({"actor": torch.randn(4, 72)}, batch_size=[4])
  groups = {"actor": ["actor"], "critic": ["actor"]}
  actor = MLPModel(
    obs,
    groups,
    "actor",
    12,
    hidden_dims=[16],
    distribution_cfg={"class_name": "GaussianDistribution"},
  )
  critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16])
  storage = RolloutStorage("rl", 4, 3, obs, [12], "cpu")
  algorithm = cls(
    actor,
    critic,
    storage,
    num_learning_epochs=2,
    num_mini_batches=2,
    learning_rate=1e-4,
    schedule="fixed",
    use_clipped_value_loss=clipped,
    normalize_advantage_per_mini_batch=normalize,
    device="cpu",
  )
  return algorithm, obs


def fill_rollout(algorithm, obs):
  torch.manual_seed(27)
  with torch.no_grad():
    for _ in range(3):
      algorithm.act(obs)
      obs = TensorDict({"actor": torch.randn(4, 72)}, batch_size=[4])
      algorithm.process_env_step(obs, torch.randn(4), torch.zeros(4).bool(), {})
    algorithm.compute_returns(obs)


@pytest.mark.parametrize("clipped", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_zero_aux_matches_upstream_ppo(clipped, normalize):
  upstream, obs = make_algorithm(PPO, clipped=clipped, normalize=normalize)
  selective, obs2 = make_algorithm(
    SelectiveTeacherPPO, clipped=clipped, normalize=normalize
  )
  selective.aux_enabled = False
  fill_rollout(upstream, obs)
  fill_rollout(selective, obs2)
  torch.manual_seed(53)
  expected = upstream.update()
  torch.manual_seed(53)
  actual = selective.update()
  for name, value in expected.items():
    assert actual[name] == pytest.approx(value, abs=1e-7)
  for left, right in (
    (upstream.actor, selective.actor),
    (upstream.critic, selective.critic),
  ):
    for name, value in left.state_dict().items():
      torch.testing.assert_close(value, right.state_dict()[name], rtol=0, atol=1e-7)


def test_aux_update_finite_frozen_teachers_and_student_actions(tmp_path):
  algorithm, obs = make_algorithm(SelectiveTeacherPPO)
  checkpoint = tmp_path / "teacher.pt"
  torch.save({"actor_state_dict": algorithm.actor.state_dict()}, checkpoint)
  command = torch.tensor([[0.2, 0.0, 0.0], [0.7, 0.0, 0.0]]).repeat(2, 1)
  term = SimpleNamespace(
    command=command, target_command=command, transition_age=torch.ones(4)
  )
  data = SimpleNamespace(
    projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(4, 1),
    root_link_pos_w=torch.ones(4, 3),
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: term),
    scene={"robot": SimpleNamespace(data=data)},
    num_envs=4,
    episode_length_buf=torch.ones(4) * 100,
    step_dt=0.02,
  )
  algorithm.bind(env, checkpoint, checkpoint, stage=1)
  before = [deepcopy(t.state_dict()) for t in algorithm.teachers]
  torch.manual_seed(77)
  with torch.no_grad():
    expected_action = algorithm.actor(obs, stochastic_output=True)
    torch.manual_seed(77)
    actual_action = algorithm.act(obs)
  torch.testing.assert_close(actual_action, expected_action, rtol=0, atol=0)
  algorithm.pending.clear()
  fill_rollout(algorithm, obs)
  result = algorithm.update()
  assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
  assert result["selective_huber"] > 0
  assert 0 < result["auxiliary_scale"] <= 1
  assert result["dagger_size"] == 12
  assert result["teacher_low_fraction"] == pytest.approx(0.5)
  assert result["teacher_omni_fraction"] == pytest.approx(0.5)
  for teacher, state in zip(algorithm.teachers, before, strict=True):
    assert not teacher.training
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    for key, value in state.items():
      torch.testing.assert_close(value, teacher.state_dict()[key], rtol=0, atol=0)


def test_low_requires_pure_current_and_target_within_point_three():
  command = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.3, 0.0, 0.0],
      [0.0, -0.3, 0.0],
      [0.0, 0.0, 0.3],
      [0.3001, 0.0, 0.0],
      [0.2, 0.1, 0.0],
      [0.2, 0.0, 0.0],
    ]
  )
  target = command.clone()
  target[-1] = torch.tensor([0.2, 0.1, 0.0])
  ids = teacher_selection(command, target, torch.ones(7).bool(), torch.ones(7))
  assert ids.tolist() == [1, 1, 1, 1, 2, 2, 2]


def test_auxiliary_gradient_is_capped_at_thirty_percent(monkeypatch):
  baseline, obs = make_algorithm(SelectiveTeacherPPO)
  auxiliary, obs2 = make_algorithm(SelectiveTeacherPPO)
  baseline.aux_enabled = False
  baseline.max_grad_norm = auxiliary.max_grad_norm = 1e8
  fill_rollout(baseline, obs)
  fill_rollout(auxiliary, obs2)
  auxiliary.dagger.add(obs2["actor"], torch.full((4, 12), 100.0), torch.ones(4).long())
  monkeypatch.setattr(auxiliary, "auxiliary_weights", lambda: (1e5, 1e5))
  captured = []

  def capture(algorithm):
    captured.append(
      torch.cat(
        [
          p.grad.detach().flatten().clone()
          for p in algorithm.actor.parameters()
          if p.grad is not None
        ]
      )
    )

  monkeypatch.setattr(baseline.optimizer, "step", lambda: capture(baseline))
  monkeypatch.setattr(auxiliary.optimizer, "step", lambda: capture(auxiliary))
  torch.manual_seed(93)
  baseline.update()
  base = captured[0]
  captured.clear()
  torch.manual_seed(93)
  auxiliary.update()
  ratio = (captured[0] - base).norm() / base.norm()
  assert ratio.item() == pytest.approx(0.3, abs=1e-5)


def test_checkpoint_resumes_stage_optimizer_and_replay_rng(tmp_path):
  algorithm, obs = make_algorithm(SelectiveTeacherPPO)
  algorithm.stage = 2
  algorithm.stage_update = 123
  algorithm.dagger = RecentDAgger(7)
  algorithm.dagger.add(obs["actor"], torch.ones(4, 12), torch.tensor([1, 2, 1, 2]))
  fill_rollout(algorithm, obs)
  algorithm.update()
  checkpoint = tmp_path / "continuation.pt"
  torch.save(algorithm.save(), checkpoint)
  restored, obs2 = make_algorithm(SelectiveTeacherPPO)
  assert restored.load(torch.load(checkpoint, weights_only=False), None, True)
  assert (restored.stage, restored.stage_update) == (2, 124)
  assert restored.dagger.capacity == 7
  for a, b in zip(
    algorithm.dagger.sample(13, "cpu"), restored.dagger.sample(13, "cpu"), strict=True
  ):
    torch.testing.assert_close(a, b, rtol=0, atol=0)
  fill_rollout(algorithm, obs)
  fill_rollout(restored, obs2)
  torch.manual_seed(101)
  expected = algorithm.update()
  torch.manual_seed(101)
  actual = restored.update()
  assert expected == actual
  for name, value in algorithm.actor.state_dict().items():
    torch.testing.assert_close(value, restored.actor.state_dict()[name], rtol=0, atol=0)
