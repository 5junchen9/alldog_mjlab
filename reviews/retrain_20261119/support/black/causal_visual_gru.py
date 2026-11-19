"""32维视觉残差记忆：两段重叠交接，保留有限的图像重算窗口。"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class State:
  episode: torch.Tensor
  frame: torch.Tensor
  capture_tick: torch.Tensor
  tick: int
  hidden: torch.Tensor
  fresh_count: torch.Tensor
  warm_hidden: torch.Tensor

  def detach(self):
    return State(
      self.episode,
      self.frame,
      self.capture_tick,
      self.tick,
      self.hidden.detach(),
      self.fresh_count,
      self.warm_hidden.detach(),
    )


class CausalVisualGRU(nn.Module):
  """每张健康新帧更新一次；旧段淡出时，新段已有半个窗口的记忆。"""

  def __init__(self, enabled=False, max_age=6, memory_frames=50):
    super().__init__()
    if type(enabled) is not bool or type(max_age) is not int or max_age < 0:
      raise ValueError("configuration")
    if type(memory_frames) is not int or memory_frames < 2 or memory_frames % 2:
      raise ValueError("memory_frames must be a positive even window >=2")
    self.enabled = enabled
    self.max_age = max_age
    self.memory_frames = memory_frames
    if enabled:
      self.gru = nn.GRUCell(32, 64)
      self.projection = nn.Linear(64, 32, bias=False)
      # Residual over the original feature; starts bitwise equal on
      # eligible frames. The first gradient reaches projection; later
      # steps also train GRU once projection is nonzero.
      nn.init.zeros_(self.projection.weight)

  def forward(
    self,
    visual,
    *,
    tick=None,
    episode=None,
    capture_episode=None,
    frame=None,
    capture_tick=None,
    mask=None,
    state=None,
  ):
    if not self.enabled:
      return visual, state
    if (
      visual.ndim != 2
      or visual.shape[1] != 32
      or not visual.is_floating_point()
      or not torch.isfinite(visual).all()
    ):
      raise ValueError("original finite float32-compatible [lane,32] feature required")
    n = visual.shape[0]
    for x in (episode, capture_episode, frame, capture_tick):
      if x.shape != (n,) or x.device != visual.device or x.dtype != torch.long:
        raise ValueError("native per-row integer frame identity")
    if mask.shape != (n,) or mask.device != visual.device or mask.dtype != torch.bool:
      raise ValueError("native frame valid mask per lane")
    present = mask & (capture_episode == episode) & (frame >= 0)
    if (
      type(tick) is not int
      or tick < 0
      or (episode < 0).any()
      or (present & ((capture_tick < 0) | (capture_tick > tick))).any()
    ):
      raise ValueError("invalid or future clock")
    valid = present & ((tick - capture_tick) <= self.max_age)
    if state is None:
      h = visual.new_zeros((n, 64))
      warm = torch.zeros_like(h)
      old = torch.full_like(episode, -1)
      old_cap = old.clone()
      count = torch.zeros_like(episode)
    else:
      if (
        state.hidden.shape != (n, 64)
        or state.warm_hidden.shape != state.hidden.shape
        or state.hidden.device != visual.device
        or state.warm_hidden.device != visual.device
        or state.hidden.dtype != visual.dtype
        or state.warm_hidden.dtype != visual.dtype
        or tick != state.tick + 1
      ):
        raise ValueError("explicit fresh state at discontinuity")
      if (episode < state.episode).any():
        raise ValueError("reversed episode serial")
      same = episode == state.episode
      h = torch.where(same[:, None], state.hidden, 0)
      warm = torch.where(same[:, None], state.warm_hidden, 0)
      old = torch.where(same, state.frame, -1)
      old_cap = torch.where(same, state.capture_tick, -1)
      count = torch.where(same, state.fresh_count, 0)
    first = old < 0
    reused = valid & ~first & (frame == old)
    if (reused & (old_cap != capture_tick)).any() or (
      valid & ~first & ~reused & ((frame < old) | (capture_tick <= old_cap))
    ).any():
      raise ValueError("stale/rekeyed frame identity")
    fresh = valid & ~reused
    # 后半段同时积累新记忆并逐帧交接，满窗时提升已预热的半窗记忆。
    half = self.memory_frames // 2
    handoff = fresh & (count >= self.memory_frames)
    h = torch.where(handoff[:, None], warm, h)
    warm = torch.where(handoff[:, None], 0.0, warm)
    count = torch.where(handoff, half, count) + fresh.long()
    indices = fresh.nonzero(as_tuple=True)[0]
    if len(indices):
      h = h.index_copy(0, indices, self.gru(visual[indices], h[indices]))
    warm_ids = (fresh & (count > half)).nonzero(as_tuple=True)[0]
    if len(warm_ids):
      warm = warm.index_copy(0, warm_ids, self.gru(visual[warm_ids], warm[warm_ids]))
    new = State(
      episode.clone(),
      torch.where(fresh, frame, old),
      torch.where(fresh, capture_tick, old_cap),
      tick,
      h,
      count,
      warm,
    )
    # Unusable frames do not expose carried activations. Keep the original
    # baseline feature for the current (possibly all-invalid) depth; no
    # positional mismatch in first call or when disabled.
    weight = ((count - half).to(visual.dtype) / half).clamp(0, 1)[:, None]
    memory = (1 - weight) * h + weight * warm
    return visual + torch.where(valid[:, None], self.projection(memory), 0), new

  def forward_sequence(self, visual, rows, state=None):
    if len(visual) != len(rows) or not len(rows):
      raise ValueError("aligned causal sequence")
    out = []
    for x, metadata in zip(visual, rows, strict=True):
      y, state = self(x, state=state, **metadata)
      out.append(y)
    return torch.stack(out), state
