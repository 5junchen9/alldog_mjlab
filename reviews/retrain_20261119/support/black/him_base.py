"""HIM学生：速度估计、对比表征和教师动作蒸馏。

依据HIMLoco的显式速度+隐式表征结构；本适配使用5×45历史。
两个回归目标使用Huber，对比目标保持HIM的原型交换交叉熵。
教师使用独立的普通PPO，不参与这里的更新。
"""

from itertools import chain

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def sinkhorn(scores: torch.Tensor) -> torch.Tensor:
  # 特征和原型均归一化，分数范围有限；减最大值保持指数计算稳定。
  assignment = torch.exp((scores - scores.max()) / 0.05).T
  prototype_count, batch_size = assignment.shape
  assignment /= assignment.sum()
  for _ in range(3):
    assignment /= assignment.sum(dim=1, keepdim=True)
    assignment /= prototype_count
    assignment /= assignment.sum(dim=0, keepdim=True)
    assignment /= batch_size
  return (assignment * batch_size).T


class HIMStudent(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(5 * 45, 128),
      nn.ELU(),
      nn.Linear(128, 64),
      nn.ELU(),
      nn.Linear(64, 3 + 16),
    )
    self.target_encoder = nn.Sequential(
      nn.Linear(45, 128),
      nn.ELU(),
      nn.Linear(128, 64),
      nn.ELU(),
      nn.Linear(64, 16),
    )
    self.prototypes = nn.Embedding(32, 16)

  def encode(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = self.encoder(history.flatten(start_dim=1))
    return encoded[:, :3], F.normalize(encoded[:, 3:], dim=-1)

  def estimator_losses(
    self,
    history: torch.Tensor,
    next_frame: torch.Tensor,
    next_velocity: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    velocity, source_latent = self.encode(history)
    target_latent = F.normalize(self.target_encoder(next_frame.detach()), dim=-1)
    with torch.no_grad():
      self.prototypes.weight.copy_(F.normalize(self.prototypes.weight, dim=-1))
    source_scores = source_latent @ self.prototypes.weight.T
    target_scores = target_latent @ self.prototypes.weight.T
    source_assignment = sinkhorn(source_scores)
    target_assignment = sinkhorn(target_scores)
    swap_loss = (
      -0.5
      * (
        source_assignment * F.log_softmax(target_scores / 3.0, dim=-1)
        + target_assignment * F.log_softmax(source_scores / 3.0, dim=-1)
      ).mean()
    )
    velocity_loss = F.huber_loss(velocity, next_velocity.detach(), delta=1.0)
    return velocity_loss, swap_loss

  def estimator_parameters(self):
    return chain(
      self.encoder.parameters(),
      self.target_encoder.parameters(),
      self.prototypes.parameters(),
    )
