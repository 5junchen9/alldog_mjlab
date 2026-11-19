"""学习器恢复的小型工具；随机状态不包含物理、相机或episode状态。"""

import random

import numpy as np
import torch

SCHEMA = "shared_actor_learner_v1"


def capture_rng(device):
  """只捕获全局RNG；CUDA始终使用显式设备，不读取默认设备。"""
  device = torch.device(device)
  if device.type == "cuda" and device.index is None:
    raise ValueError("CUDA RNG requires an explicit device index")
  numpy_state = np.random.get_state(legacy=True)
  assert isinstance(numpy_state, tuple), "NumPy RNG requires legacy MT19937 state"
  return {
    "python": random.getstate(),
    "numpy": (
      numpy_state[0],
      torch.from_numpy(numpy_state[1].copy()),
      int(numpy_state[2]),
      int(numpy_state[3]),
      float(numpy_state[4]),
    ),
    "torch_cpu": torch.get_rng_state().clone(),
    "cuda": (
      torch.cuda.get_rng_state(device).cpu().clone()
      if device.type == "cuda" and torch.cuda.is_available()
      else None
    ),
  }


def restore_rng(packet, device):
  """恢复学习器随机序列；调用方仍需显式重开物理episode。"""
  device = torch.device(device)
  if device.type == "cuda" and device.index is None:
    raise ValueError("CUDA RNG requires an explicit device index")
  if packet["cuda"] is not None and device.type != "cuda":
    raise ValueError("saved CUDA RNG requires a CUDA target device")
  if packet["cuda"] is None and device.type == "cuda":
    raise ValueError("saved CUDA RNG is missing for CUDA target device")
  random.setstate(packet["python"])
  numpy_state = packet["numpy"]
  np.random.set_state(
    (
      numpy_state[0],
      numpy_state[1].cpu().numpy(),
      numpy_state[2],
      numpy_state[3],
      numpy_state[4],
    )
  )
  torch.set_rng_state(packet["torch_cpu"].cpu())
  if packet["cuda"] is not None:
    torch.cuda.set_rng_state(packet["cuda"].cpu(), device)


def _same_recipe(saved, expected):
  # 配方中的True、1、1.0不能因Python普通相等判断而混为一项。
  if type(saved) is not type(expected):
    return False
  if isinstance(saved, dict):
    return saved.keys() == expected.keys() and all(
      _same_recipe(value, expected[key]) for key, value in saved.items()
    )
  if isinstance(saved, (list, tuple)):
    return len(saved) == len(expected) and all(
      _same_recipe(left, right) for left, right in zip(saved, expected, strict=True)
    )
  return saved == expected


def validate_resume(packet, phase, recipe):
  """恢复只接受相同阶段和配方，不替换模型输入语义或推断训练资格。"""
  if packet.get("phase") != phase:
    raise ValueError("resume phase mismatch")
  learning_state = packet.get("learning_state")
  if not isinstance(learning_state, dict) or learning_state.get("schema") != SCHEMA:
    raise ValueError("learning_state schema mismatch")
  if not isinstance(recipe, dict) or not _same_recipe(
    learning_state.get("recipe"), recipe
  ):
    raise ValueError("resume recipe mismatch")
  updates = packet.get("completed_updates")
  if type(updates) is not int or updates < 1:
    raise ValueError("completed_updates must be an integer >=1")
  calls = learning_state.get("optimizer_calls")
  if type(calls) is not int or calls < 0:
    raise ValueError("optimizer_calls must be an integer >=0")
  return learning_state


def parameter_signature(named_parameters):
  """按实际optimizer绑定顺序记录名称、形状和dtype，不重新排序。"""
  return [
    [name, list(parameter.shape), str(parameter.dtype)]
    for name, parameter in named_parameters
  ]
