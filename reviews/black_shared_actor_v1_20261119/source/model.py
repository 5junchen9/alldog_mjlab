"""共享动作头的新模型身份；旧模型只提供初始化权重。"""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

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

MODEL_KIND = "black_shared_actor_v1"


class FrozenHIM(nn.Module):
  """沿用旧HIM：预测下一步速度3维，以及单位化的隐特征16维。"""

  def __init__(self, encoder):
    super().__init__()
    self.encoder = encoder
    self.requires_grad_(False)
    self.eval()

  def train(self, mode=True):
    # 外层teacher.train()也不能启用HIM的训练状态。
    return super().train(False)

  def forward(self, history):
    with torch.no_grad():
      encoded = self.encoder(history.flatten(start_dim=1))
      next_velocity = encoded[:, :3]
      latent = F.normalize(encoded[:, 3:], dim=-1)
      return torch.cat((next_velocity, latent), dim=-1)


class Teacher(nn.Module):
  """p45+冻结HIM19+可见地形32；完整446维状态不进入actor。"""

  def __init__(self, him, actor, terrain_dim=768):
    super().__init__()
    self.him = him
    self.actor = actor
    self.terrain_dim = terrain_dim
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


def make_modules():
  """只创建模型结构；新bundle加载无需读取原r800文件。"""
  frontend = DepthHIMStudent()
  him = FrozenHIM(frontend.encoder)
  gru = CausalVisualGRU(enabled=True)
  return him, frontend.actor, frontend.depth_encoder, gru


def make_initial_modules(learner):
  """严格读取r800模型权重；不恢复旧Adam、旧训练身份或循环状态。"""
  frontend = DepthHIMStudent()
  weights = dict(learner["frontend"])
  weights.update({"actor." + name: value for name, value in learner["head"].items()})
  frontend.load_state_dict(weights, strict=True)

  gru = CausalVisualGRU(enabled=True)
  gru.load_state_dict(learner["gru"], strict=True)
  him = FrozenHIM(frontend.encoder)
  return him, frontend.actor, frontend.depth_encoder, gru
