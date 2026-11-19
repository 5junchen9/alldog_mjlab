"""小型合成rollout验证原生PPO真更新，不代表机器人训练或能力验收。"""

import pytest
import torch
from model import Teacher, make_modules
from ppo import OnnxTeacher, build_ppo
from tensordict import TensorDict


def setup_ppo():
  him, actor, _, _ = make_modules()
  teacher = Teacher(him, actor)
  critic = torch.randn(4, 446)
  critic[:, 259:] = 1
  obs = TensorDict(
    {"history": torch.randn(4, 225), "geometry": torch.randn(4, 768), "critic": critic},
    batch_size=[4],
  )
  algorithm = build_ppo(
    obs,
    teacher,
    num_steps=4,
    ppo_options={
      "num_learning_epochs": 1,
      "num_mini_batches": 2,
      "schedule": "fixed",
      "learning_rate": 1e-4,
    },
  )
  return teacher, algorithm, obs


def test_native_ppo_reencodes_geometry_and_updates_only_allowed_parameters():
  torch.manual_seed(743)
  teacher, algorithm, obs = setup_ppo()
  assert algorithm.actor.mlp is teacher.actor
  assert algorithm.actor.policy is teacher
  optimizer_ids = {
    id(p) for group in algorithm.optimizer.param_groups for p in group["params"]
  }
  expected_ids = {
    id(p)
    for module in (algorithm.actor, algorithm.critic)
    for p in module.parameters()
    if p.requires_grad
  }
  assert optimizer_ids == expected_ids
  assert not optimizer_ids & {id(p) for p in teacher.him.parameters()}
  assert not algorithm.optimizer.state

  before = {name: p.detach().clone() for name, p in teacher.named_parameters()}
  critic_before = {
    name: p.detach().clone() for name, p in algorithm.critic.named_parameters()
  }
  gradient_modes = []
  hook = teacher.terrain_encoder.register_forward_pre_hook(
    lambda module, args: gradient_modes.append(torch.is_grad_enabled())
  )
  with torch.no_grad():
    for tick in range(4):
      actions = algorithm.act(obs)
      rewards = -actions.square().mean(dim=-1) + torch.arange(4) * 0.1
      next_obs = obs.clone()
      next_obs["history"] = obs["history"] + 0.001
      next_obs["geometry"] = obs["geometry"] + (tick + 1) * 0.002
      algorithm.process_env_step(
        next_obs, rewards, torch.zeros(4, dtype=torch.bool), {}
      )
      obs = next_obs
    algorithm.compute_returns(obs)

  assert algorithm.storage.observations["geometry"].shape == (4, 4, 768)
  losses = algorithm.update()
  hook.remove()
  assert gradient_modes == [False] * 4 + [True] * 2
  assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
  assert algorithm.optimizer.state
  # 原生PPO在update末尾更新SplitNormalizer，统计随新模型重新建立。
  normalizer = algorithm.critic.obs_normalizer.state_dict()
  assert normalizer["raw72.count"].item() == 16
  assert normalizer["terrain.count"].item() == 16
  after = dict(teacher.named_parameters())
  for prefix in ("actor.", "terrain_encoder."):
    assert any(
      not torch.equal(before[name], after[name])
      for name in before
      if name.startswith(prefix)
    )
  assert all(
    torch.equal(before[name], after[name]) for name in before if name.startswith("him.")
  )
  assert any(
    not torch.equal(value, dict(algorithm.critic.named_parameters())[name])
    for name, value in critic_before.items()
  )


def test_actor_cannot_read_critic_and_wrapper_preserves_initial_head():
  teacher, algorithm, obs = setup_ppo()
  expected = teacher(obs["history"].reshape(-1, 5, 45), obs["geometry"])
  assert torch.equal(algorithm.actor(obs), expected)
  altered = obs.clone()
  altered["critic"] = torch.randn_like(obs["critic"]) * 1000
  assert torch.equal(algorithm.actor(altered), expected)
  teacher.freeze()
  with pytest.raises(ValueError, match="trainable"):
    build_ppo(obs, teacher, num_steps=4)


def test_timeout_uses_terminal_next_value_and_failure_wins():
  _, algorithm, obs = setup_ppo()
  with torch.no_grad():
    algorithm.act(obs)
    terminal = obs["critic"].clone()
    terminal[:, :72] += 3
    reset_obs = obs.clone()
    reset_obs["critic"][:, :72] += 100
    target = TensorDict({"critic": terminal[:1]}, batch_size=[1])
    value = algorithm.critic(target).item()
    assert algorithm.transition.values is not None
    assert abs(value - algorithm.transition.values[0].item()) > 1e-6
    algorithm.process_env_step(
      reset_obs,
      torch.ones(4),
      torch.tensor([1, 1, 1, 0]),
      {
        "time_outs": torch.tensor([True, True, False, False]),
        "terminated": torch.tensor([False, True, True, False]),
        "terminal_critic": terminal,
      },
    )
  expected = torch.ones(4)
  expected[0] += algorithm.gamma * value
  torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], expected)


def test_critic_initialization_does_not_depend_on_environment_rng_consumption():
  teacher, first, obs = setup_ppo()
  torch.rand(1003)
  second = build_ppo(obs, teacher, num_steps=4)
  for name, value in first.critic.state_dict().items():
    torch.testing.assert_close(value, second.critic.state_dict()[name], rtol=0, atol=0)


def test_rsl_torchscript_and_onnx_export_preserve_teacher_mean(tmp_path):
  import numpy as np
  import onnxruntime

  teacher, algorithm, obs = setup_ppo()
  history, geometry = obs["history"].reshape(-1, 5, 45), obs["geometry"]
  expected = teacher(history, geometry).detach()
  scripted = torch.jit.script(algorithm.actor.as_jit())
  script_path = tmp_path / "teacher.ts"
  scripted.save(str(script_path))
  torch.testing.assert_close(
    torch.jit.load(str(script_path))(history, geometry), expected, rtol=0, atol=0
  )
  wrapper = algorithm.actor.as_onnx(False)
  assert isinstance(wrapper, OnnxTeacher)
  onnx_path = tmp_path / "teacher.onnx"
  torch.onnx.export(
    wrapper,
    wrapper.get_dummy_inputs(),
    str(onnx_path),
    input_names=wrapper.input_names,
    output_names=wrapper.output_names,
    dynamic_axes={
      "history": {0: "batch"},
      "geometry": {0: "batch"},
      "actions": {0: "batch"},
    },
    opset_version=18,
    dynamo=False,
  )
  session = onnxruntime.InferenceSession(
    str(onnx_path), providers=["CPUExecutionProvider"]
  )
  actual = session.run(
    None, {"history": history.numpy(), "geometry": geometry.numpy()}
  )[0]
  assert isinstance(actual, np.ndarray)
  torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-5, atol=1e-6)
