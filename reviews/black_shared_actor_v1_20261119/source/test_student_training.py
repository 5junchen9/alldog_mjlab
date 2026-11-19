"""真实Torch模型+合成物理；验证采样/更新接线，不证明机器人能力。"""

import copy
from types import SimpleNamespace

import pytest
import student_training as training
import torch
from checkpoint import load_bundle
from learner_state import capture_rng, restore_rng
from model import Teacher, make_modules
from student import SharedStudent


class FakeFrames:
  def __init__(self, count, invalid_from=None):
    self.tick = 0
    self.episode = torch.zeros(count, dtype=torch.long)
    self.birth = torch.zeros_like(self.episode)
    self.capture_tick = torch.zeros_like(self.episode)
    self.frame = torch.arange(count)
    self.depth = torch.full((count, 48, 64, 1), 1.2)
    self.geometry = torch.full((count, 768), 0.3)
    self.invalid_from = invalid_from
    if invalid_from == 0:
      self.depth.zero_()
      self.geometry.zero_()

  def metadata(self):
    return dict(
      tick=self.tick,
      episode=self.episode.clone(),
      capture_episode=self.episode.clone(),
      frame=self.frame.clone(),
      capture_tick=self.capture_tick.clone(),
      mask=torch.full_like(
        self.episode,
        self.invalid_from is None or self.tick < self.invalid_from,
        dtype=torch.bool,
      ),
    )

  def advance(self):
    self.tick += 1
    if self.tick == 17:
      self.episode[0] += 1
      self.birth[0] = self.tick
    due = (self.tick - self.birth) % 5 == 0
    self.capture_tick[due] = self.tick
    self.frame[due] += len(self.episode)
    self.depth[due] = 1.2 + self.tick / 100
    self.geometry[due] = self.depth[due, 0, 0, 0, None] / 4
    if self.invalid_from is not None and self.tick >= self.invalid_from:
      self.depth.zero_()
      self.geometry.zero_()


class FakeEnv:
  def __init__(self, count=2, invalid_from=None):
    self.count, self.device = count, "cpu"
    self.metrics = SimpleNamespace(policy_update=0)
    self.frames = FakeFrames(count, invalid_from)
    self.history = torch.arange(count * 225).reshape(count, 5, 45).float() / 1000
    self.actions, self.visited = [], []
    self.expected_action = None
    self.bind()

  def bind(self):
    self.obs = dict(
      history=self.history.flatten(1).clone(),
      geometry=self.frames.geometry.clone(),
      critic=torch.zeros(self.count, 446),
    )

  def step(self, action):
    if self.expected_action is not None:
      torch.testing.assert_close(action, self.expected_action, rtol=0, atol=0)
    self.actions.append(action.clone())
    self.visited.append((self.history.clone(), self.obs["geometry"].clone()))
    # 合成运动确实依赖学生动作，使之后访问的状态由学生控制产生。
    next_proprio = self.history[:, -1].clone()
    next_proprio[:, :12] += action * 0.02
    next_proprio[:, 12] += 0.02
    self.history = torch.cat((self.history[:, 1:], next_proprio[:, None]), dim=1)
    self.frames.advance()
    if self.frames.tick == 17:
      self.history[0] = 0
    self.bind()


@pytest.fixture
def modules():
  threads = torch.get_num_threads()
  torch.set_num_threads(1)
  torch.manual_seed(19)
  him, actor, cnn, gru = make_modules()
  yield Teacher(him, actor), cnn, gru
  torch.set_num_threads(threads)


def test_student_controls_and_teacher_labels_its_states_across_rounds(
  modules,
  tmp_path,
  monkeypatch,
):
  teacher, cnn, gru = modules
  before = {name: value.clone() for name, value in teacher.state_dict().items()}
  cnn_before = {name: value.clone() for name, value in cnn.state_dict().items()}
  env = FakeEnv()
  queries, acting, refreshed = [], {}, {}
  original_features = teacher.features

  def features(history, geometry):
    queries.append((history.clone(), geometry.clone()))
    return original_features(history, geometry)

  monkeypatch.setattr(teacher, "features", features)
  original_forward = SharedStudent.forward

  def forward(self, history, depth, metadata, state=None):
    result = original_forward(self, history, depth, metadata, state)
    if not torch.is_grad_enabled():
      if metadata["tick"] == env.frames.tick:
        env.expected_action = result[0].clone()
        acting[metadata["tick"]] = (state, result[2])
      else:
        refreshed[metadata["tick"]] = result[2]
    return result

  monkeypatch.setattr(SharedStudent, "forward", forward)
  result = training.train_student(env, teacher, cnn, gru, 2, tmp_path)
  assert len(env.actions) == result["control_steps"] == 48
  assert result["environment_transitions"] == 96
  assert result["completed_updates"] == 2 and result["final_state_tick"] == 47
  assert env.metrics.policy_update == 1  # 第二轮采集开始时已有一次更新。
  expected_history = torch.stack([row[0] for row in env.visited]).flatten(0, 1)
  expected_geometry = torch.stack([row[1] for row in env.visited]).flatten(0, 1)
  torch.testing.assert_close(torch.cat([row[0] for row in queries]), expected_history)
  torch.testing.assert_close(torch.cat([row[1] for row in queries]), expected_geometry)
  assert acting[0][0] is None
  assert acting[24][0].tick == 23
  assert bool(acting[24][0].hidden.any())
  torch.testing.assert_close(acting[24][0].hidden, refreshed[23].hidden)
  assert not torch.equal(acting[24][0].hidden, acting[23][1].hidden)
  assert acting[17][1].episode.tolist() == [1, 0]
  assert all(
    torch.equal(value, teacher.state_dict()[name]) for name, value in before.items()
  )
  assert all(parameter.grad is None for parameter in teacher.parameters())
  assert any(
    not torch.equal(value, cnn.state_dict()[name]) for name, value in cnn_before.items()
  )
  saved = torch.load(tmp_path / "student.pt", weights_only=True)
  assert saved["phase"] == "student"
  assert saved["completed_updates"] == saved["learning_state"]["optimizer_calls"] == 2
  assert saved["learning_state"]["schema"] == "shared_actor_learner_v1"
  assert saved["qualification"] == "not_evaluated"
  assert not saved["exact_physical_resume"]
  optimizer = saved["learning_state"]["optimizer"]
  assert all(int(value["step"]) == 2 for value in optimizer["state"].values())
  assert len(optimizer["state"]) == len(list(cnn.parameters()) + list(gru.parameters()))
  assert not result["recipe"]["exact_long_episode_replay"]
  assert result["recipe"]["gradient_clip_norm"] == 1.0
  assert result["recipe"]["num_envs"] == env.count
  assert result["starting_updates"] == 0 and result["added_updates"] == 2
  assert not result["learner_resumed"]


def test_partial_microbatch_has_lane_weight_and_one_optimizer_step(
  modules,
  tmp_path,
  monkeypatch,
):
  teacher, cnn, gru = modules
  copies = copy.deepcopy((teacher, cnn, gru))
  monkeypatch.setattr(training, "LANES_PER_BATCH", 3)
  full = training.train_student(FakeEnv(3), teacher, cnn, gru, 1, tmp_path / "full")
  monkeypatch.setattr(training, "LANES_PER_BATCH", 2)
  split = training.train_student(FakeEnv(3), *copies, 1, tmp_path / "split")
  assert full["round_metrics"][0]["loss"] == pytest.approx(
    split["round_metrics"][0]["loss"],
    rel=1e-6,
  )
  packets = [torch.load(row["checkpoint"], weights_only=True) for row in (full, split)]
  full_state, split_state = [
    row["learning_state"]["optimizer"]["state"] for row in packets
  ]
  for key, value in full_state.items():
    assert int(value["step"]) == int(split_state[key]["step"]) == 1
    torch.testing.assert_close(
      value["exp_avg"],
      split_state[key]["exp_avg"],
      rtol=1e-4,
      atol=2e-9,
    )


def test_nonfinite_gradient_stops_before_adam_and_preserves_weights(
  modules,
  tmp_path,
  monkeypatch,
):
  teacher, cnn, gru = modules
  before = [copy.deepcopy(module.state_dict()) for module in modules]
  adam_calls = []
  original_step = torch.optim.Adam.step

  def step(self, *args, **kwargs):
    adam_calls.append(True)
    return original_step(self, *args, **kwargs)

  monkeypatch.setattr(torch.optim.Adam, "step", step)
  handle = cnn[0].weight.register_hook(
    lambda gradient: torch.full_like(gradient, float("nan"))
  )
  env = FakeEnv()
  try:
    with pytest.raises(RuntimeError, match="non-finite"):
      training.train_student(env, teacher, cnn, gru, 1, tmp_path)
  finally:
    handle.remove()
  assert len(env.actions) == 24
  assert adam_calls == []
  assert not (tmp_path / "student.pt").exists()
  for module, saved in zip(modules, before, strict=True):
    assert all(
      torch.equal(value, module.state_dict()[name]) for name, value in saved.items()
    )


def test_all_invalid_window_trains_cnn_without_initializing_gru_cell_adam(
  modules, tmp_path
):
  teacher, cnn, gru = modules
  cnn_before = copy.deepcopy(cnn.state_dict())
  cell_before = copy.deepcopy(gru.gru.state_dict())
  result = training.train_student(
    FakeEnv(invalid_from=0), teacher, cnn, gru, 1, tmp_path
  )
  assert not result["round_metrics"][0]["gru_cell_used"]
  assert any(
    not torch.equal(value, cnn.state_dict()[name]) for name, value in cnn_before.items()
  )
  assert all(
    torch.equal(value, gru.gru.state_dict()[name])
    for name, value in cell_before.items()
  )
  assert all(parameter.grad is None for parameter in gru.gru.parameters())
  saved = torch.load(tmp_path / "student.pt", weights_only=True)
  optimizer = saved["learning_state"]["optimizer"]
  cell_parameters = set(gru.gru.parameters())
  parameters = list(cnn.parameters()) + list(gru.parameters())
  identifiers = optimizer["param_groups"][0]["params"]
  for parameter, identifier in zip(parameters, identifiers, strict=True):
    if parameter in cell_parameters:
      assert identifier not in optimizer["state"]


def test_invalid_window_preserves_existing_gru_cell_weights_and_adam_moments(
  modules, tmp_path, monkeypatch
):
  teacher, cnn, gru = modules
  # 首轮就产生非零GRUCell梯度与动量，避免只检查空/零Adam状态。
  with torch.no_grad():
    gru.projection.weight.normal_(std=0.02)
  snapshots = []
  original_step = torch.optim.Adam.step

  def record_step(optimizer, *args, **kwargs):
    result = original_step(optimizer, *args, **kwargs)
    snapshots.append(
      {
        "cnn": copy.deepcopy(cnn.state_dict()),
        "cell": copy.deepcopy(gru.gru.state_dict()),
        "adam": [
          copy.deepcopy(optimizer.state[parameter])
          for parameter in gru.gru.parameters()
        ],
      }
    )
    return result

  monkeypatch.setattr(torch.optim.Adam, "step", record_step)
  result = training.train_student(
    FakeEnv(invalid_from=24), teacher, cnn, gru, 2, tmp_path
  )
  assert [row["gru_cell_used"] for row in result["round_metrics"]] == [True, False]
  assert len(snapshots) == 2
  first, second = snapshots
  assert any(bool(slot["exp_avg"].any()) for slot in first["adam"])
  assert all(
    torch.equal(value, second["cell"][name]) for name, value in first["cell"].items()
  )
  for old, new in zip(first["adam"], second["adam"], strict=True):
    assert int(old["step"]) == int(new["step"]) == 1
    for name in old:
      assert torch.equal(old[name], new[name])
  assert any(
    not torch.equal(value, second["cnn"][name]) for name, value in first["cnn"].items()
  )


class RoundResetFrames(FakeFrames):
  """每24步全部重开episode；相对帧龄相同，绝对时钟保持连续。"""

  def advance(self):
    self.tick += 1
    local_tick = self.tick % training.CONTROL_STEPS
    if local_tick == 0:
      self.episode += 1
      self.birth.fill_(self.tick)
      self.capture_tick.fill_(self.tick)
      self.frame += len(self.episode)
      self.depth.fill_(1.2)
      self.geometry.fill_(0.3)
      return
    if local_tick == 17:
      self.episode[0] += 1
      self.birth[0] = self.tick
    due = (self.tick - self.birth) % 5 == 0
    self.capture_tick[due] = self.tick
    self.frame[due] += len(self.episode)
    self.depth[due] = 1.2 + local_tick / 100
    self.geometry[due] = self.depth[due, 0, 0, 0, None] / 4


class RoundResetEnv(FakeEnv):
  """把物理重开边界对齐，单独检验学习器数值恢复，不声称续接物理。"""

  def __init__(self):
    super().__init__()
    self.frames = RoundResetFrames(self.count)
    self.initial_history = self.history.clone()
    self.bind()

  def step(self, action):
    super().step(action)
    local_tick = self.frames.tick % training.CONTROL_STEPS
    if local_tick == 0:
      self.history = self.initial_history.clone()
    else:
      # 验证恢复的Torch随机状态实际参与下一轮，而非只保存字段。
      self.history[:, -1, 13] += torch.randn(self.count) * 0.01
      if local_tick == 17:
        self.history[0] = 0
    self.bind()


def test_resume_matches_continuous_learner_at_aligned_cold_episode_boundary(
  modules, tmp_path, monkeypatch
):
  teacher, cnn, gru = modules
  initial_teacher, initial_cnn, initial_gru = copy.deepcopy(modules)
  initial_rng = capture_rng("cpu")
  full = training.train_student(
    RoundResetEnv(), teacher, cnn, gru, 2, tmp_path / "continuous", save_every=1
  )
  restore_rng(initial_rng, "cpu")
  first = training.train_student(
    RoundResetEnv(),
    initial_teacher,
    initial_cnn,
    initial_gru,
    1,
    tmp_path / "first",
    save_every=1,
  )
  teacher, cnn, gru, packet = load_bundle(first["checkpoint"])
  torch.rand(13)  # 加载模型和其他调用消耗RNG后，resume仍须回到保存点。
  collected_prefixes = []
  original_collect = training._collect

  def collect(env, student, state):
    collected_prefixes.append(state)
    return original_collect(env, student, state)

  monkeypatch.setattr(training, "_collect", collect)
  env = RoundResetEnv()
  resumed = training.train_student(
    env, teacher, cnn, gru, 1, tmp_path / "resumed", resume=packet, save_every=2
  )
  assert collected_prefixes == [None]
  assert resumed["completed_updates"] == resumed["optimizer_calls"] == 2
  assert resumed["starting_updates"] == 1 and resumed["added_updates"] == 1
  assert resumed["control_steps"] == 24
  assert resumed["round_metrics"][0]["round"] == 2
  assert resumed["learner_resumed"] and env.metrics.policy_update == 1
  assert resumed["cold_start"] == ["physics", "curriculum", "camera", "GRU"]
  assert not resumed["exact_physical_resume"]
  continuous_packet = torch.load(full["checkpoint"], weights_only=True)
  resumed_packet = torch.load(resumed["checkpoint"], weights_only=True)
  for name in ("teacher", "cnn", "gru"):
    for key, value in continuous_packet[name].items():
      assert torch.equal(value, resumed_packet[name][key])
  continuous_state = continuous_packet["learning_state"]
  resumed_state = resumed_packet["learning_state"]
  assert (
    resumed_state["restart_boundary"] == "new_physics_curriculum_camera_gru_episode"
  )
  assert (
    continuous_state["optimizer"]["param_groups"]
    == resumed_state["optimizer"]["param_groups"]
  )
  for identifier, slots in continuous_state["optimizer"]["state"].items():
    for name, value in slots.items():
      assert torch.equal(value, resumed_state["optimizer"]["state"][identifier][name])
  assert torch.equal(
    continuous_state["rng"]["torch_cpu"], resumed_state["rng"]["torch_cpu"]
  )
  assert (tmp_path / "first" / "student_update_000001.pt").is_file()
  assert (tmp_path / "resumed" / "student_update_000002.pt").is_file()


def test_resume_rejects_parameter_order_before_loading_optimizer_or_sampling(
  modules, tmp_path, monkeypatch
):
  teacher, cnn, gru = modules
  first = training.train_student(FakeEnv(), teacher, cnn, gru, 1, tmp_path / "first")
  teacher, cnn, gru, packet = load_bundle(first["checkpoint"])
  packet["learning_state"]["parameter_signature"].reverse()
  load_calls = []
  monkeypatch.setattr(
    torch.optim.Adam, "load_state_dict", lambda *args: load_calls.append(True)
  )
  env = FakeEnv()
  with pytest.raises(ValueError, match="parameter signature mismatch"):
    training.train_student(
      env, teacher, cnn, gru, 1, tmp_path / "rejected", resume=packet
    )
  assert load_calls == [] and env.actions == []
  assert not list((tmp_path / "rejected").glob("*.pt"))


def test_resume_rejects_changed_frozen_teacher_before_sampling(modules, tmp_path):
  teacher, cnn, gru = modules
  first = training.train_student(FakeEnv(), teacher, cnn, gru, 1, tmp_path / "first")
  teacher, cnn, gru, packet = load_bundle(first["checkpoint"])
  with torch.no_grad():
    next(teacher.terrain_encoder.parameters()).add_(0.01)
  env = FakeEnv()
  with pytest.raises(ValueError, match="teacher identity mismatch"):
    training.train_student(
      env, teacher, cnn, gru, 1, tmp_path / "rejected", resume=packet
    )
  assert env.actions == []
  assert not list((tmp_path / "rejected").glob("*.pt"))


@pytest.mark.parametrize("count, feature_weight", [(3, 1.0), (2, 0.5)])
def test_resume_rejects_changed_student_recipe(
  modules, tmp_path, count, feature_weight
):
  teacher, cnn, gru = modules
  first = training.train_student(FakeEnv(), teacher, cnn, gru, 1, tmp_path / "first")
  teacher, cnn, gru, packet = load_bundle(first["checkpoint"])
  env = FakeEnv(count)
  with pytest.raises(ValueError, match="recipe mismatch"):
    training.train_student(
      env, teacher, cnn, gru, 1, tmp_path / "rejected", feature_weight, resume=packet
    )
  assert env.actions == []


def test_periodic_checkpoints_retain_each_saved_update(modules, tmp_path):
  teacher, cnn, gru = modules
  result = training.train_student(
    FakeEnv(), teacher, cnn, gru, 3, tmp_path, save_every=1
  )
  paths = [tmp_path / f"student_update_{number:06d}.pt" for number in (1, 2, 3)]
  assert result["periodic_checkpoints"] == [str(path) for path in paths]
  saved = [torch.load(path, weights_only=True) for path in paths]
  for number, packet in enumerate(saved, start=1):
    assert packet["completed_updates"] == number
    assert packet["learning_state"]["optimizer_calls"] == number
    assert all(
      int(value["step"]) == number
      for value in packet["learning_state"]["optimizer"]["state"].values()
    )
  assert not torch.equal(saved[0]["cnn"]["0.weight"], saved[2]["cnn"]["0.weight"])
  final = torch.load(tmp_path / "student.pt", weights_only=True)
  assert final["completed_updates"] == 3
  assert torch.equal(final["cnn"]["0.weight"], saved[2]["cnn"]["0.weight"])
  assert not list(tmp_path.glob("*.pending"))
