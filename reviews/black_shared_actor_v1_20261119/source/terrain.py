"""Visible geometry from the student's exact metric depth pixels.

MuJoCo-Warp supplies planar Z in metres (not Euclidean ray distance). We keep
actual source pixels rather than average/interpolate across surface boundaries.
Geometry uses optical camera axes: x right, y down, z forward. This is a sparse
visible point set, not a complete height map or a hidden terrain query.
"""

import torch
from torch import Tensor

RAW_HEIGHT, RAW_WIDTH = 90, 160
DEPTH_HEIGHT, DEPTH_WIDTH = 48, 64
POINT_HEIGHT, POINT_WIDTH = 12, 16
GEOMETRY_DIM = POINT_HEIGHT * POINT_WIDTH * 4
DEPTH_SCALE_M = 4.0


def _nearest_pixels(source_size: int, target_size: int, device) -> Tensor:
  """Nearest source pixel to each target pixel centre; ties choose the right."""
  positions = torch.arange(target_size, device=device)
  return ((2 * positions + 1) * source_size) // (2 * target_size)


def reduce_depth(raw: Tensor) -> Tensor:
  """Select [B,48,64,1] from FP32 metric [B,90,160,1]; no inpainting.

  The returned tensor owns its pixels, because the renderer reuses its buffer.
  Misses/nonfinite values are retained for the existing student's validity mask.
  Fixed subsampling can miss thin obstacles; it does not add unseen information.
  """
  if raw.dtype != torch.float32 or raw.shape[1:] != (RAW_HEIGHT, RAW_WIDTH, 1):
    raise ValueError("expected FP32 metric depth [B,90,160,1]")
  rows = _nearest_pixels(RAW_HEIGHT, DEPTH_HEIGHT, raw.device)
  cols = _nearest_pixels(RAW_WIDTH, DEPTH_WIDTH, raw.device)
  return raw[:, rows[:, None], cols[None, :], :]


def camera_frustum(sensor_size: Tensor, intrinsics: Tensor) -> Tensor:
  """MuJoCo perspective [left,right,bottom,top] divided by the near distance.

  sensor_size is [2]; intrinsics is [B,4] in MuJoCo's physical sensor units
  [fx,fy,cx,cy]. Match render_util.compute_ray's aspect correction at 160x90.
  """
  aspect = RAW_WIDTH / RAW_HEIGHT
  sensor_width = torch.minimum(sensor_size[0], sensor_size[1] * aspect)
  sensor_height = torch.minimum(sensor_size[1], sensor_size[0] / aspect)
  fx, fy, cx, cy = intrinsics.unbind(-1)
  left = (-sensor_width / 2 + cx) / fx
  right = (sensor_width / 2 + cx) / fx
  bottom = (-sensor_height / 2 - cy) / fy
  top = (sensor_height / 2 - cy) / fy
  return torch.stack((left, right, bottom, top), dim=-1)


def renderer_frustum(renderer, env_ids: Tensor) -> Tensor:
  """Read the direct160 adapter's owned parameters for these global env IDs.

  A leading intrinsic dimension of one is shared by all worlds. Otherwise,
  select the actual world IDs, including reordered/subset renderer chunks.
  """
  if (renderer.width, renderer.height) != (RAW_WIDTH, RAW_HEIGHT):
    raise ValueError("projection requires the direct160 renderer")
  sensor_sizes, intrinsics = renderer._direct160_owned[:2]
  camera_id = renderer.camera_id
  selected = intrinsics[:, camera_id]
  if len(selected) == 1:
    selected = selected.expand(len(env_ids), -1)
  else:
    selected = selected[env_ids.to(device=selected.device, dtype=torch.long)]
  return camera_frustum(sensor_sizes[camera_id], selected)


def geometry(depth: Tensor, bounds: Tensor) -> Tensor:
  """Project the shared depth64 into [B,768]: row-major (x/4,y/4,z/4,valid).

  bounds is [4] or [B,4], from renderer_frustum. Each of the 12x16 points uses
  its original 160x90 source pixel's ray, not an approximate resized intrinsic.
  Zero, negative and nonfinite depth give (0,0,0,0). Four metres is a scale,
  not a clipping distance, matching preprocess_raw_depth's existing contract.
  The caller owns same-episode, capture-time, age and fresh-frame alignment.
  """
  if depth.dtype != torch.float32 or depth.shape[1:] != (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    1,
  ):
    raise ValueError("expected shared FP32 metric depth [B,48,64,1]")
  rows = _nearest_pixels(DEPTH_HEIGHT, POINT_HEIGHT, depth.device)
  cols = _nearest_pixels(DEPTH_WIDTH, POINT_WIDTH, depth.device)
  source_rows = _nearest_pixels(RAW_HEIGHT, DEPTH_HEIGHT, depth.device)[rows]
  source_cols = _nearest_pixels(RAW_WIDTH, DEPTH_WIDTH, depth.device)[cols]
  z = depth[:, rows[:, None], cols[None, :], 0]
  valid = torch.isfinite(z) & (z > 0)
  z = torch.where(valid, z, 0.0)

  bounds = bounds.to(device=depth.device, dtype=depth.dtype)
  left, right, bottom, top = bounds.reshape(-1, 4).unbind(-1)
  u = (source_cols.to(depth.dtype) + 0.5) / RAW_WIDTH
  v = (source_rows.to(depth.dtype) + 0.5) / RAW_HEIGHT
  ray_x = left[:, None] + (right - left)[:, None] * u[None, :]
  ray_y = -(top[:, None] + (bottom - top)[:, None] * v[None, :])
  x = z * ray_x[:, None, :]
  y = z * ray_y[:, :, None]
  points = torch.stack(
    (x / DEPTH_SCALE_M, y / DEPTH_SCALE_M, z / DEPTH_SCALE_M, valid), dim=-1
  )
  return points.reshape(len(depth), GEOMETRY_DIM)
