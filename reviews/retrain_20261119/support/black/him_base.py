"""HIM估计器：去命令本体历史速度估计与稳定SwAV对比表征。

依据HIMLoco的显式速度+隐式表征结构；本适配使用5×45历史并剔除3维command，
仅从5×42纯本体动力学历史预测下一帧速度(3)和单位隐变量(16)。
"""

from itertools import chain

import torch
from torch import nn
from torch.nn import functional as F

COMMAND_SLICE = slice(30, 33)
PROPRIO_NO_CMD_DIM = 42
TEMPERATURE = 0.5


def strip_command(frames: torch.Tensor) -> torch.Tensor:
  """从45维p45帧（[B,45]或[B,T,45]）剔除30:33处的3维command，保留42维本体。"""
  return torch.cat((frames[..., :30], frames[..., 33:]), dim=-1)


@torch.no_grad()
def sinkhorn(scores: torch.Tensor) -> torch.Tensor:
  # 特征和原型均归一化，减最大值与分母clamp_min防止溢出或除零NaN。
  assignment = torch.exp((scores - scores.max()) / 0.05).T
  prototype_count, batch_size = assignment.shape
  assignment /= assignment.sum().clamp_min(1e-8)
  for _ in range(3):
    assignment /= assignment.sum(dim=1, keepdim=True).clamp_min(1e-8)
    assignment /= prototype_count
    assignment /= assignment.sum(dim=0, keepdim=True).clamp_min(1e-8)
    assignment /= batch_size
  return (assignment * batch_size).T


class HIMStudent(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(5 * PROPRIO_NO_CMD_DIM, 128),
      nn.ELU(),
      nn.Linear(128, 64),
      nn.ELU(),
      nn.Linear(64, 3 + 16),
    )
    self.target_encoder = nn.Sequential(
      nn.Linear(PROPRIO_NO_CMD_DIM + 3, 128),
      nn.ELU(),
      nn.Linear(128, 64),
      nn.ELU(),
      nn.Linear(64, 16),
    )
    self.prototypes = nn.Embedding(32, 16)

  def encode(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    proprio = strip_command(history).flatten(start_dim=1)
    encoded = self.encoder(proprio)
    return encoded[:, :3], F.normalize(encoded[:, 3:], dim=-1)

  @torch.no_grad()
  def normalize_prototypes(self) -> None:
    self.prototypes.weight.copy_(F.normalize(self.prototypes.weight, dim=-1))

  def estimator_losses(
    self,
    history: torch.Tensor,
    next_frame: torch.Tensor,
    next_velocity: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    velocity, source_latent = self.encode(history)
    target_input = torch.cat(
      (strip_command(next_frame.detach()), next_velocity.detach()), dim=-1
    )
    target_latent = F.normalize(self.target_encoder(target_input), dim=-1)
    # 不在微批前向中原地修改prototypes.weight；由update_him在step后统一归一化。
    proto = F.normalize(self.prototypes.weight, dim=-1)
    source_scores = source_latent @ proto.T
    target_scores = target_latent @ proto.T
    source_assignment = sinkhorn(source_scores)
    target_assignment = sinkhorn(target_scores)
    swap_loss = -0.5 * (
      (source_assignment * F.log_softmax(target_scores / TEMPERATURE, dim=-1))
      .sum(dim=-1)
      .mean()
      + (target_assignment * F.log_softmax(source_scores / TEMPERATURE, dim=-1))
      .sum(dim=-1)
      .mean()
    )
    if len(history) == 1:
      # 单条样本只训练速度；对比匹配没有跨样本信息，仍保留零梯度路径。
      swap_loss = swap_loss * 0.0
    velocity_loss = F.huber_loss(velocity, next_velocity.detach(), delta=1.0)
    return velocity_loss, swap_loss

  def estimator_parameters(self):
    return chain(
      self.encoder.parameters(),
      self.target_encoder.parameters(),
      self.prototypes.parameters(),
    )
