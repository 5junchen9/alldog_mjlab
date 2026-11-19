"""Isolated direct160 renderer adapter + source-only reducer. No live mutations."""

import copy
import dataclasses

import numpy as np
import torch


def frustum(sensor, intrinsic, wh):
  sw, sh = map(float, sensor)
  fx, fy, cx, cy = map(float, intrinsic)
  w, h = wh
  if sh <= 0 or sw <= 0 or fx <= 0 or fy <= 0:
    raise ValueError("explicit perspective intrinsics required")
  if w / h > sw / sh:
    sh = sw / (w / h)
  elif w / h < sw / sh:
    sw = sh * (w / h)
  return np.array(
    [
      (-sw * 0.5 + cx) / fx,
      (sw * 0.5 + cx) / fx,
      (-sh * 0.5 - cy) / fy,
      (sh * 0.5 - cy) / fy,
    ]
  )  # left,right,bottom,top


def retarget(sensor, intrinsic, old_wh=(424, 240), new_wh=(160, 90)):
  left, right, bottom, top = frustum(sensor, intrinsic, old_wh)
  sw, sh = new_wh[0] / new_wh[1], 1.0
  fx, fy = sw / (right - left), sh / (top - bottom)
  return np.array([sw, sh]), np.array(
    [fx, fy, fx * (left + right) / 2, -fy * (top + bottom) / 2]
  )


def rays(sensor, intrinsic, wh):
  left, right, bottom, top = frustum(sensor, intrinsic, wh)
  w, h = wh
  u = (np.arange(w) + 0.5) / w
  v = (np.arange(h) + 0.5) / h
  x, y = np.meshgrid(left + (right - left) * u, top + (bottom - top) * v)
  d = np.stack((x, y, -np.ones_like(x)), -1)
  return d / np.linalg.norm(d, axis=-1, keepdims=True)


def make_renderer(*, renderer_class, mjm, model, data, camera_id=0, chunk=32):
  """Call ONLY in approved idle probe. Private camera render copies, no sim writes.
  Actual v7 ChunkRenderer supplied by owner. Source mjm/model/data are not mutated.
  """
  if chunk not in (32, 64, 128):
    raise ValueError("explicit tested probe chunks")
  import warp as wp

  if int(mjm.cam_projection[camera_id]) != 0:
    raise ValueError("perspective required")
  private = copy.deepcopy(mjm)
  sensor0 = wp.to_torch(model.cam_sensorsize).detach().clone()
  intrinsic0 = wp.to_torch(model.cam_intrinsic).detach().clone()
  sensor = sensor0[camera_id].cpu().numpy()
  converted = [retarget(sensor, i.cpu().numpy()) for i in intrinsic0[:, camera_id]]
  sensor0[camera_id] = torch.as_tensor(
    converted[0][0], dtype=sensor0.dtype, device=sensor0.device
  )
  for k, (_, intr) in enumerate(converted):
    intrinsic0[k, camera_id] = torch.as_tensor(
      intr, dtype=intrinsic0.dtype, device=intrinsic0.device
    )
  private.cam_sensorsize[camera_id] = converted[0][0]
  private.cam_intrinsic[camera_id] = converted[0][1]
  private.cam_resolution[camera_id] = [160, 90]
  render_model = dataclasses.replace(
    model,
    cam_sensorsize=wp.from_torch(sensor0, dtype=model.cam_sensorsize.dtype),
    cam_intrinsic=wp.from_torch(intrinsic0, dtype=model.cam_intrinsic.dtype),
  )
  renderer = renderer_class(
    private,
    render_model,
    data,
    chunk_size=chunk,
    width=160,
    height=90,
    camera_id=camera_id,
  )
  renderer._direct160_owned = (sensor0, intrinsic0, private, render_model)
  return (
    renderer,
    None,
    {
      "reference_wh": [424, 240],
      "render_wh": [160, 90],
      "camera_id": camera_id,
      "chunk": chunk,
      "frustum_retargeted": True,
      "model_mutated": False,
    },
  )
