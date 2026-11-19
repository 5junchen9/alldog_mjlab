"""Bounded full-resolution depth rendering; no simulation stepping or pixel reduction."""

import dataclasses

import mujoco_warp as mw
import warp as wp

DATA_FIELDS = (
  "geom_xpos",
  "geom_xmat",
  "cam_xpos",
  "cam_xmat",
  "light_xpos",
  "light_xdir",
  "flexvert_xpos",
)
MODEL_FIELDS = (
  "geom_dataid",
  "geom_matid",
  "geom_size",
  "geom_rgba",
  "cam_fovy",
  "cam_intrinsic",
  "light_type",
  "light_castshadow",
  "light_active",
  "light_attenuation",
  "light_cutoff",
  "light_exponent",
  "light_ambient",
  "light_diffuse",
  "light_specular",
  "mat_texid",
  "mat_texrepeat",
  "mat_emission",
  "mat_specular",
  "mat_shininess",
  "mat_rgba",
)


class ChunkRenderer:
  """Caller must hold simulation state fixed until iterator is exhausted.

  Returned depth aliases the reusable context: consume/copy before next yield.
  No flex support. Context is separate from simulation's raycast context.
  """

  def __init__(
    self, mjm, model, data, chunk_size=32, width=1280, height=720, camera_id=0
  ):
    if model.nflex:
      raise ValueError("Flex rendering is not implemented")
    if not 0 <= camera_id < mjm.ncam or chunk_size < 1:
      raise ValueError("Invalid camera or chunk size")
    self.mjm, self.model, self.data = mjm, model, data
    self.chunk_size, self.width, self.height = chunk_size, width, height
    self.camera_id, self.contexts = camera_id, {}

  def _context(self, count):
    if count not in self.contexts:
      self.contexts[count] = mw.create_render_context(
        self.mjm,
        nworld=count,
        cam_res=(self.width, self.height),
        cam_active=[i == self.camera_id for i in range(self.mjm.ncam)],
        render_rgb=False,
        render_depth=True,
        render_seg=False,
        use_textures=False,
        use_shadows=False,
        use_precomputed_rays=False,
      )
    return self.contexts[count]

  def _slice(self, a, start, end, shared):
    if shared and a.shape[0] == 1:
      return a
    if a.shape[0] != self.data.nworld:
      raise ValueError(f"Unexpected world axis {a.shape}, expected {self.data.nworld}")
    # Tensor and Warp views retain original storage; no source mutation.
    return wp.from_torch(wp.to_torch(a)[start:end], dtype=a.dtype)

  def frames(self):
    for start in range(0, self.data.nworld, self.chunk_size):
      end = min(start + self.chunk_size, self.data.nworld)
      d = dataclasses.replace(
        self.data,
        nworld=end - start,
        **{
          k: self._slice(getattr(self.data, k), start, end, False) for k in DATA_FIELDS
        },
      )
      m = dataclasses.replace(
        self.model,
        **{
          k: self._slice(getattr(self.model, k), start, end, True) for k in MODEL_FIELDS
        },
      )
      rc = self._context(end - start)
      mw.refit_bvh(m, d, rc)
      mw.render(m, d, rc)
      yield (
        start,
        end,
        wp.to_torch(rc.depth_data).view(end - start, self.height, self.width, 1),
      )

  def frames_for(self, env_ids):
    """Render ordered arbitrary worlds only; reset refresh must not affect others.

    Yields (ids, depth). Output aliases reusable context as in frames().
    Empty requests yield nothing. Source simulation/model remain untouched.
    """
    import torch

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

    def gather(a, part, shared):
      if shared and a.shape[0] == 1:
        return a
      if a.shape[0] != self.data.nworld:
        raise ValueError(f"Unexpected world axis {a.shape}")
      return wp.from_torch(
        wp.to_torch(a).index_select(0, part).contiguous(), dtype=a.dtype
      )

    for start in range(0, ids.numel(), self.chunk_size):
      part = ids[start : start + self.chunk_size]
      d = dataclasses.replace(
        self.data,
        nworld=part.numel(),
        **{k: gather(getattr(self.data, k), part, False) for k in DATA_FIELDS},
      )
      m = dataclasses.replace(
        self.model,
        **{k: gather(getattr(self.model, k), part, True) for k in MODEL_FIELDS},
      )
      rc = self._context(part.numel())
      mw.refit_bvh(m, d, rc)
      mw.render(m, d, rc)
      yield (
        part,
        wp.to_torch(rc.depth_data).view(part.numel(), self.height, self.width, 1),
      )
