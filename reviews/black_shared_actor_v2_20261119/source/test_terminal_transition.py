"""末帧监督不被局部reset覆盖，也不推进存活lane的输入。"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from runtime import AlignedFrames, NativeEnv
from tensordict import TensorDict


def fake_env():
  env: Any = object.__new__(NativeEnv)
  env.count = 3
  env.evaluation = None
  env.frames = AlignedFrames(3, "cpu")
  native = TensorDict(
    {
      "history": torch.arange(675.0).reshape(3, 5, 45),
      "critic": torch.arange(1338.0).reshape(3, 446),
    },
    [3],
  )
  reward = torch.tensor([1.0, 2.0, 3.0])
  terminated = torch.tensor([True, False, False])
  timeout = torch.tensor([True, True, False])
  done = (terminated | timeout).long()
  env.wrapper = SimpleNamespace(step=lambda _: (native, reward, done, {}))
  reset_ids = []

  def reset(*, env_ids):
    reset_ids.extend(env_ids.tolist())
    terminated.zero_()
    timeout.zero_()
    reward.zero_()
    reset_obs = {}
    for key, value in native.items():
      assert isinstance(value, torch.Tensor)
      reset_obs[key] = torch.full_like(value, -7.0)
    return reset_obs, {}

  env.plant = SimpleNamespace(
    reset_terminated=terminated,
    reset_time_outs=timeout,
    reset=reset,
    recorder_manager=SimpleNamespace(record_pre_reset=lambda _: None),
  )
  env.bind = lambda value: value
  return env, native.clone(), reset_ids


def test_terminal_targets_survive_reset_and_failure_wins():
  env, original, reset_ids = fake_env()
  obs, reward, done, info = env.step(torch.zeros(3, 12))
  assert reset_ids == [0, 1]
  assert info["time_outs"].tolist() == [False, True, False]
  assert info["terminated"].tolist() == [True, False, False]
  torch.testing.assert_close(info["terminal_history"], original["history"])
  torch.testing.assert_close(info["terminal_critic"], original["critic"])
  torch.testing.assert_close(info["next_velocity"], original["critic"][:, :3])
  assert reward.tolist() == [1.0, 2.0, 3.0]
  assert done.tolist() == [1, 1, 0]
  assert env.frames.episode.tolist() == [1, 1, 0]
  for key in obs.keys():
    torch.testing.assert_close(obs[key][2], original[key][2])
    assert bool((obs[key][:2] == -7).all())


def test_nonfinite_action_never_steps_physics():
  env, _, reset_ids = fake_env()
  env.wrapper.step = lambda _: pytest.fail("physics must not step")
  with pytest.raises(FloatingPointError):
    env.step(torch.full((3, 12), float("nan")))
  assert reset_ids == []
