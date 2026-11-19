"""Isolated GRU ablation for Black's existing 32D visual payload; off by default."""

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

  def detach(self):
    return State(
      self.episode, self.frame, self.capture_tick, self.tick, self.hidden.detach()
    )


class CausalVisualGRU(nn.Module):
  """One update per valid fresh frame. No privileged inputs, no official wiring."""

  def __init__(self, enabled=False, max_age=6):
    super().__init__()
    if type(enabled) is not bool or type(max_age) is not int or max_age < 0:
      raise ValueError("configuration")
    self.enabled = enabled
    self.max_age = max_age
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
      old = torch.full_like(episode, -1)
      old_cap = old.clone()
    else:
      if (
        state.hidden.shape != (n, 64)
        or state.hidden.device != visual.device
        or state.hidden.dtype != visual.dtype
        or tick != state.tick + 1
      ):
        raise ValueError("explicit fresh state at discontinuity")
      if (episode < state.episode).any():
        raise ValueError("reversed episode serial")
      same = episode == state.episode
      h = torch.where(same[:, None], state.hidden, 0)
      old = torch.where(same, state.frame, -1)
      old_cap = torch.where(same, state.capture_tick, -1)
    first = old < 0
    reused = valid & ~first & (frame == old)
    if (reused & (old_cap != capture_tick)).any() or (
      valid & ~first & ~reused & ((frame < old) | (capture_tick <= old_cap))
    ).any():
      raise ValueError("stale/rekeyed frame identity")
    fresh = valid & ~reused
    indices = fresh.nonzero(as_tuple=True)[0]
    if len(indices):
      h = h.index_copy(0, indices, self.gru(visual[indices], h[indices]))
    new = State(
      episode.clone(),
      torch.where(fresh, frame, old),
      torch.where(fresh, capture_tick, old_cap),
      tick,
      h,
    )
    # Unusable frames do not expose carried activations. Keep the original
    # baseline feature for the current (possibly all-invalid) depth; no
    # positional mismatch in first call or when disabled.
    return visual + torch.where(valid[:, None], self.projection(h), 0), new

  def forward_sequence(self, visual, rows, state=None):
    if len(visual) != len(rows) or not len(rows):
      raise ValueError("aligned causal sequence")
    out = []
    for x, metadata in zip(visual, rows, strict=True):
      y, state = self(x, state=state, **metadata)
      out.append(y)
    return torch.stack(out), state
