"""深度与有效掩码的视觉编码器；动作由教师和学生共用的actor生成。"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DepthContract:
  width: int = 160
  height: int = 120
  min_m: float = 0.2
  max_m: float = 4.0
  max_age_s: float = 0.12

  def __post_init__(self):
    if (
      min(self.width, self.height) <= 0
      or not 0 < self.min_m < self.max_m
      or self.max_age_s <= 0
    ):
      raise ValueError("invalid depth contract")


DEFAULT_DEPTH_CONTRACT = DepthContract()


def make_depth_encoder():
  """只创建32维视觉编码器，不创建任何动作头。"""
  return nn.Sequential(
    nn.Conv2d(2, 16, 5, 2),
    nn.ELU(),
    nn.Conv2d(16, 32, 3, 2),
    nn.ELU(),
    nn.Conv2d(32, 32, 3, 2),
    nn.ELU(),
    nn.AdaptiveAvgPool2d((2, 2)),
    nn.Flatten(),
    nn.Linear(128, 32),
    nn.ELU(),
  )


def preprocess_raw_depth(
  depth_m, capture_s, control_s, same_episode, cfg=DEFAULT_DEPTH_CONTRACT
):
  """Preserve all finite positive raw depth; max_m is scale, NOT range clipping."""
  if depth_m.ndim != 4 or depth_m.shape[1:] != (cfg.height, cfg.width, 1):
    raise ValueError("depth must match declared resolution")
  if depth_m.dtype != torch.float32:
    raise ValueError("raw depth must be float32")
  b = depth_m.shape[0]
  if any(x.shape != (b,) for x in (capture_s, control_s, same_episode)):
    raise ValueError("per-environment timestamps/episode metadata required")
  if same_episode.dtype != torch.bool:
    raise ValueError("same_episode must be bool")
  if not (torch.isfinite(capture_s).all() and torch.isfinite(control_s).all()):
    raise ValueError("nonfinite timestamps")
  age = control_s - capture_s
  if (age < -1e-6).any():
    raise ValueError("future camera frame forbidden")
  valid = torch.isfinite(depth_m) & (depth_m > 0)
  valid = valid & ((age <= cfg.max_age_s) & same_episode)[:, None, None, None]
  value = torch.where(valid, depth_m / cfg.max_m, torch.zeros_like(depth_m))
  return torch.cat((value, valid.to(value.dtype)), -1).permute(0, 3, 1, 2).contiguous()
