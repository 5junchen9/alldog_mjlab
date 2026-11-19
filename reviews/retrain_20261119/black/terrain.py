"""Visible geometry from the student's exact metric depth pixels.

MuJoCo-Warp supplies planar Z in metres (not Euclidean ray distance). We keep
actual source pixels rather than average/interpolate across surface boundaries.
Both reductions retain the nearest valid visible point and its actual source
pixel. Geometry uses optical camera axes: x right, y down, z forward. Near noise
can win this selection; it is neither a noise filter nor a complete height map.
"""

import torch
from torch import Tensor
from torch.nn import functional as F

RAW_HEIGHT, RAW_WIDTH = 90, 160
DEPTH_HEIGHT, DEPTH_WIDTH = 48, 64
POINT_HEIGHT, POINT_WIDTH = 12, 16
GEOMETRY_DIM = POINT_HEIGHT * POINT_WIDTH * 4
DEPTH_SCALE_M = 4.0
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 4.0


def reduce_depth(raw: Tensor) -> tuple[Tensor, Tensor]:
  """Return metric depth64 and int64 original row*160+column per output bin.

  Adaptive bins cover all raw pixels (edge bins may overlap). Each bin chooses
  its nearest finite positive sample; all-invalid bins return depth=0,pixel=-1.
  This aggregation does not invent a surface at an originally invalid ray:
  downstream projection always uses the selected valid ray's stored pixel.
  """
  if raw.dtype != torch.float32 or raw.shape[1:] != (RAW_HEIGHT, RAW_WIDTH, 1):
    raise ValueError("expected FP32 metric depth [B,90,160,1]")
  valid = torch.isfinite(raw) & (raw >= DEPTH_MIN_M) & (raw <= DEPTH_MAX_M)
  score = torch.where(valid, -raw, -torch.inf).permute(0, 3, 1, 2)
  closest, pixels = F.adaptive_max_pool2d(
    score, (DEPTH_HEIGHT, DEPTH_WIDTH), return_indices=True
  )
  valid = torch.isfinite(closest)
  depth = torch.where(valid, -closest, 0.0).permute(0, 2, 3, 1).contiguous()
  pixels = torch.where(valid, pixels, -1).squeeze(1)
  return depth, pixels


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


def geometry(depth: Tensor, pixels: Tensor, bounds: Tensor) -> Tensor:
  """Project the shared depth64 into [B,768]: row-major (x/4,y/4,z/4,valid).

  Each 4x4 depth64 bin contributes its closest valid point, preserving the
  first reduction's original 160x90 ray. Thus its nearest thin hit is retained
  even when it falls between the former fixed 12x16 sampling locations.
  bounds is [4] or [B,4], from renderer_frustum.
  Depth outside [0.2,4.0]m gives (0,0,0,0), matching preprocess_raw_depth.
  The caller owns same-episode, capture-time, age and fresh-frame alignment.
  """
  if depth.dtype != torch.float32 or depth.shape[1:] != (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    1,
  ):
    raise ValueError("expected shared FP32 metric depth [B,48,64,1]")
  if pixels.shape != depth.shape[:3] or pixels.dtype != torch.long:
    raise ValueError("original source pixel index required for every depth64 bin")
  valid = (
    torch.isfinite(depth[..., 0])
    & (depth[..., 0] >= DEPTH_MIN_M)
    & (depth[..., 0] <= DEPTH_MAX_M)
  )
  if bool((valid & ((pixels < 0) | (pixels >= RAW_HEIGHT * RAW_WIDTH))).any()):
    raise ValueError("valid depth requires a real original source pixel")
  score = torch.where(valid, -depth[..., 0], -torch.inf).unsqueeze(1)
  closest, selected = F.max_pool2d(score, 4, return_indices=True)
  z = -closest.squeeze(1)
  source = pixels.flatten(1).gather(1, selected.flatten(1)).reshape_as(z)
  valid = torch.isfinite(z)
  z = torch.where(valid, z, 0.0)
  source_rows = source // RAW_WIDTH
  source_cols = source % RAW_WIDTH

  bounds = bounds.to(device=depth.device, dtype=depth.dtype)
  left, right, bottom, top = bounds.reshape(-1, 4).unbind(-1)
  u = (source_cols.to(depth.dtype) + 0.5) / RAW_WIDTH
  v = (source_rows.to(depth.dtype) + 0.5) / RAW_HEIGHT
  ray_x = left[:, None, None] + (right - left)[:, None, None] * u
  ray_y = -(top[:, None, None] + (bottom - top)[:, None, None] * v)
  x = z * ray_x
  y = z * ray_y
  points = torch.stack(
    (x / DEPTH_SCALE_M, y / DEPTH_SCALE_M, z / DEPTH_SCALE_M, valid), dim=-1
  )
  return points.reshape(len(depth), GEOMETRY_DIM)
