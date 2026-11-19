"""Single-context, fixed-staging renderer. No sim changes or precision reduction."""

import dataclasses

import mujoco_warp as mw
import torch
import warp as wp

from support.black.chunk_renderer import (
  DATA_FIELDS,
  MODEL_FIELDS,
)
from support.black.chunk_renderer import (
  ChunkRenderer as BaseRenderer,
)


class ChunkRenderer(BaseRenderer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._staging = {}

  @property
  def context_count(self):
    return len(self.contexts)

  def _gather_fixed(self, field, a, ids, shared):
    if shared and a.shape[0] == 1:
      return a
    if a.shape[0] != self.data.nworld:
      raise ValueError(f"Unexpected world axis for {field}: {a.shape}")
    source = wp.to_torch(a)
    shape = (self.chunk_size, *source.shape[1:])
    if field not in self._staging:
      self._staging[field] = torch.empty(
        shape, device=source.device, dtype=source.dtype
      )
    buf = self._staging[field]
    if tuple(buf.shape) != shape or buf.dtype != source.dtype:
      raise ValueError(f"Changed render schema for {field}; recreate renderer")
    torch.index_select(source, 0, ids, out=buf)
    return wp.from_torch(buf, dtype=a.dtype)

  def _render_ids(self, ids):
    actual = ids.numel()
    if not 0 < actual <= self.chunk_size:
      raise ValueError("Invalid block length")
    padded = (
      ids
      if actual == self.chunk_size
      else torch.cat((ids, ids[:1].expand(self.chunk_size - actual)))
    )
    d = dataclasses.replace(
      self.data,
      nworld=self.chunk_size,
      **{
        k: self._gather_fixed("data." + k, getattr(self.data, k), padded, False)
        for k in DATA_FIELDS
      },
    )
    m = dataclasses.replace(
      self.model,
      **{
        k: self._gather_fixed("model." + k, getattr(self.model, k), padded, True)
        for k in MODEL_FIELDS
      },
    )
    rc = self._context(self.chunk_size)
    mw.refit_bvh(m, d, rc)
    mw.render(m, d, rc)
    return wp.to_torch(rc.depth_data).view(self.chunk_size, self.height, self.width, 1)[
      :actual
    ]

  def frames(self):
    device = wp.to_torch(self.data.geom_xpos).device
    for start in range(0, self.data.nworld, self.chunk_size):
      end = min(start + self.chunk_size, self.data.nworld)
      yield start, end, self._render_ids(torch.arange(start, end, device=device))

  def frames_for(self, env_ids):
    device = wp.to_torch(self.data.geom_xpos).device
    ids = torch.as_tensor(env_ids, device=device)
    if ids.ndim != 1:
      raise ValueError("env_ids must be one-dimensional")
    if ids.numel() == 0:
      return
    if ids.dtype not in (torch.int32, torch.int64):
      raise ValueError("env_ids must be integers")
    ids = ids.to(torch.int64)
    if bool((ids < 0).any()) or bool((ids >= self.data.nworld).any()):
      raise ValueError("env_ids out of bounds")
    for start in range(0, ids.numel(), self.chunk_size):
      part = ids[start : start + self.chunk_size]
      yield part, self._render_ids(part)
