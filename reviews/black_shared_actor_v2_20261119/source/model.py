"""全部从零的共享动作头；旧文件仅复用结构与HIM损失定义。"""

import sys
from pathlib import Path

import torch
from torch import nn

LEGACY_STUDENT = (
  Path(__file__).resolve().parents[1] / "black_reliability_20261119/student"
)
sys.path.append(str(LEGACY_STUDENT))

from causal_visual_gru import CausalVisualGRU  # noqa: E402
from depth_him import (  # noqa: E402
  DepthContract as DepthContract,
)
from depth_him import DepthHIMStudent  # noqa: E402
from depth_him import (  # noqa: E402
  preprocess_raw_depth as preprocess_raw_depth,
)
from him_base import HIMStudent  # noqa: E402

MODEL_KIND = "black_shared_actor_v2"


class HIM(HIMStudent):
  """同一HIM对象交替冻结/学习，辅助目标保持next_velocity与原型交换损失。"""

  def __init__(self):
    super().__init__()
    del self.actor
    self.freeze()

  def freeze(self):
    self.requires_grad_(False)
    super().train(False)
    for parameter in self.parameters():
      parameter.grad = None
    return self

  def train(self, mode=True):
    # policy.train()不能偷偷打开估计器；辅助更新显式调用enable_learning。
    return super().train(False)

  def enable_learning(self):
    self.requires_grad_(True)
    super().train(True)

  def forward(self, history):
    # 动作/PPO损失不能通过HIM反传；estimator_losses直接调用可微encode。
    with torch.no_grad():
      next_velocity, latent = self.encode(history)
      return torch.cat((next_velocity, latent), dim=-1)


class Teacher(nn.Module):
  """p45+冻结HIM19+可见地形32；完整446维状态不进入actor。"""

  def __init__(self, him, actor, terrain_dim=768):
    super().__init__()
    self.him = him
    self.actor = actor
    self.terrain_dim = terrain_dim
    self.initialization = {
      "mode": "scratch",
      "seed": torch.initial_seed(),
      "critic_seed": torch.initial_seed() + 1,
      "modules": {
        name: "random" for name in ("HIM", "actor", "terrain_encoder", "CNN", "GRU")
      },
    }
    self.terrain_encoder = nn.Sequential(
      nn.Linear(terrain_dim, 128),
      nn.ELU(),
      nn.Linear(128, 32),
    )

  def features(self, history, geometry):
    if history.ndim != 3 or history.shape[1:] != (5, 45):
      raise ValueError("history must be [batch,5,45]")
    if geometry.shape != (len(history), self.terrain_dim):
      raise ValueError("geometry must match the declared visible grid")
    return torch.cat(
      (
        history[:, -1].detach(),
        self.him(history),
        self.terrain_encoder(geometry),
      ),
      dim=-1,
    )

  def forward(self, history, geometry):
    return self.actor(self.features(history, geometry))

  def freeze(self):
    self.eval().requires_grad_(False)
    for parameter in self.parameters():
      parameter.grad = None
    return self


def make_student_modules():
  """学生首次蒸馏重新建立视觉参数，不改变共享teacher/HIM。"""
  frontend = DepthHIMStudent()
  return frontend.depth_encoder, CausalVisualGRU(enabled=True)


def make_modules():
  """只创建随机结构；默认训练与bundle恢复均不读取任何旧权重。"""
  frontend = DepthHIMStudent()
  return HIM(), frontend.actor, frontend.depth_encoder, CausalVisualGRU(enabled=True)
