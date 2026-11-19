"""CPU-only checks of exact visible pixel geometry; no renderer or simulation."""

from types import SimpleNamespace

import pytest
import torch
from terrain import camera_frustum, geometry, reduce_depth, renderer_frustum


def test_reduce_keeps_exact_pixels_and_owns_renderer_buffer():
  raw = torch.arange(90 * 160, dtype=torch.float32).reshape(1, 90, 160, 1)
  reduced = reduce_depth(raw)
  expected = torch.tensor(
    [
      [int((y + 0.5) * 90 / 48) * 160 + int((x + 0.5) * 160 / 64) for x in range(64)]
      for y in range(48)
    ],
    dtype=torch.float32,
  )
  torch.testing.assert_close(reduced[0, :, :, 0], expected)
  raw.zero_()
  torch.testing.assert_close(reduced[0, :, :, 0], expected)


def test_projection_uses_original_pixel_centres_and_planar_z():
  depth = torch.full((1, 48, 64, 1), 2.0)
  points = geometry(depth, torch.tensor([-1.0, 1.0, -0.5, 0.5])).view(12, 16, 4)
  # Point[0,0] selects depth64 pixel(y=2,x=2), original pixel(y=4,x=6).
  expected_x = 2.0 * (-1.0 + 2.0 * 6.5 / 160) / 4
  expected_y = -2.0 * (0.5 - 4.5 / 90) / 4
  torch.testing.assert_close(
    points[0, 0], torch.tensor([expected_x, expected_y, 0.5, 1])
  )
  assert points[-1, -1, 0] > 0 and points[-1, -1, 1] > 0
  # Planar Z stays 2m even off-axis; no range-to-Z conversion is appropriate.
  assert torch.equal(points[..., 2], torch.full((12, 16), 0.5))


def test_invalid_points_zero_and_depth_scale_does_not_clip():
  depth = torch.full((1, 48, 64, 1), 8.0)
  depth[0, 2, [2, 6, 10, 14, 18], 0] = torch.tensor(
    [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
  )
  points = geometry(depth, torch.tensor([-1.0, 1.0, -1.0, 1.0])).view(12, 16, 4)
  assert torch.isfinite(points).all()
  assert torch.equal(points[0, :5], torch.zeros(5, 4))
  torch.testing.assert_close(points[0, 5, 2:], torch.tensor([2.0, 1.0]))


def test_camera_frustum_matches_renderer_formula_with_aspect_correction():
  sensor = torch.tensor([4.0, 3.0])  # Renderer crops vertical sensor extent.
  intrinsics = torch.tensor([[2.0, 3.0, 0.2, -0.1], [4.0, 2.0, -0.3, 0.2]])
  expected = torch.tensor(
    [
      [(-2 + cx) / fx, (2 + cx) / fx, (-1.125 - cy) / fy, (1.125 - cy) / fy]
      for fx, fy, cx, cy in intrinsics.tolist()
    ]
  )
  torch.testing.assert_close(camera_frustum(sensor, intrinsics), expected)


@pytest.mark.parametrize("shared", [False, True])
def test_renderer_parameters_follow_actual_global_world_ids(shared):
  sensor_sizes = torch.tensor([[1.0, 1.0], [16.0 / 9, 1.0]])
  intrinsics = torch.tensor(
    [[[1.0, 1.0, 0.0, 0.0], [1.0 + i, 2.0 + i, 0.1 * i, -0.1 * i]] for i in range(3)]
  )
  if shared:
    intrinsics = intrinsics[:1]
  renderer = SimpleNamespace(
    width=160,
    height=90,
    camera_id=1,
    _direct160_owned=(sensor_sizes, intrinsics, None, None),
  )
  env_ids = torch.tensor([2, 0])
  actual = renderer_frustum(renderer, env_ids)
  selected = intrinsics[0:1, 1].expand(2, -1) if shared else intrinsics[env_ids, 1]
  expected = camera_frustum(sensor_sizes[1], selected)
  torch.testing.assert_close(actual, expected)
  projected = geometry(torch.ones(2, 48, 64, 1), actual)
  assert projected.shape == (2, 768)
  assert torch.equal(projected[0], projected[1]) == shared


def test_reduce_and_geometry_preserve_first_hit_without_filling_holes():
  raw = torch.full((1, 90, 160, 1), 3.0)
  raw[0, 4, 6, 0] = 1.0  # Actual nearest visible hit at the selected ray.
  raw[0, 4, 16, 0] = 0.0  # A miss is not filled from its valid neighbours.
  points = geometry(reduce_depth(raw), torch.tensor([-1.0, 1.0, -1.0, 1.0]))
  points = points.view(12, 16, 4)
  torch.testing.assert_close(points[0, 0, 2:], torch.tensor([0.25, 1.0]))
  assert torch.equal(points[0, 1], torch.zeros(4))
