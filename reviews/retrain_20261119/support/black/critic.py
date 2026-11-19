"""446维critic的分段在线归一化，不含旧教师或蒸馏缓存。"""

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization

from support.black.observations import RAW72, WIDTH

SCHEMA = "raw72_height187_mask187_v1"


class SplitNormalizer(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.raw72 = EmpiricalNormalization(RAW72)
    self.terrain = EmpiricalNormalization(WIDTH - RAW72)

  def forward(self, value):
    return torch.cat(
      (self.raw72(value[..., :RAW72]), self.terrain(value[..., RAW72:])), -1
    )

  def update(self, value):
    self.raw72.update(value[..., :RAW72])
    self.terrain.update(value[..., RAW72:])


class TerrainModel(MLPModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.obs_dim != WIDTH or not self.obs_normalization:
      raise ValueError("Terrain model requires446 and online normalization")
    self.obs_normalizer = SplitNormalizer()
