"""Strict episode-boundary validation for the opt-in predictive model."""

import torch

from .spec import HIMInterface, HIMSpec


def validated_terminal_extras(
  dones: torch.Tensor, extras: dict, interface: HIMInterface, spec: HIMSpec
) -> dict:
  """Prevent predictive targets from crossing auto-reset episode boundaries."""
  done_ids = dones.reshape(-1).nonzero(as_tuple=False).flatten()
  keys = (
    interface.terminal_ids_key,
    interface.terminal_frame_key,
    interface.terminal_velocity_key,
  )
  if not done_ids.numel():
    # A stale terminal payload must not overwrite a live successor frame.
    return {key: value for key, value in extras.items() if key not in keys}
  if any(key not in extras for key in keys):
    raise ValueError("Every done environment requires HIM pre-reset terminal targets")
  ids, frame, velocity = (extras[key] for key in keys)
  if ids.ndim != 1 or ids.dtype != torch.long:
    raise ValueError("Terminal IDs must be a 1-D int64 tensor")
  if not torch.equal(ids.sort().values.to(done_ids.device), done_ids):
    raise ValueError("Terminal IDs must exactly cover done environments once")
  if frame.shape != (ids.numel(), spec.single_frame_dim):
    raise ValueError("Invalid pre-reset terminal frame shape")
  if velocity.shape != (ids.numel(), spec.velocity_dim):
    raise ValueError("Invalid pre-reset terminal velocity shape")
  if not torch.isfinite(frame).all() or not torch.isfinite(velocity).all():
    raise ValueError("Pre-reset terminal targets must be finite")
  return extras
