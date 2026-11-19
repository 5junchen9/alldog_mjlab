"""CPU-only checks of exact visible pixel geometry; no renderer or simulation."""

from types import SimpleNamespace

import pytest
import torch
from terrain import camera_frustum, geometry, reduce_depth, renderer_frustum


def test_reduce_keeps_nearest_valid_source_pixels_and_owns_renderer_buffer():
  raw = (torch.arange(90 * 160, dtype=torch.float32) + 1).reshape(1, 90, 160, 1)
  depth, pixels = reduce_depth(raw)
  expected = torch.tensor(
    [[(y * 90 // 48) * 160 + x * 160 // 64 for x in range(64)] for y in range(48)]
  )
  assert torch.equal(pixels[0], expected)
  torch.testing.assert_close(depth[0, ..., 0], expected.float() + 1)
  raw.zero_()
  torch.testing.assert_close(depth[0, ..., 0], expected.float() + 1)


def test_projection_uses_selected_original_pixel_ray_and_planar_z():
  raw = torch.full((1, 90, 160, 1), 3.0)
  raw[0, 5, 7, 0] = 1.0
  depth, pixels = reduce_depth(raw)
  points = geometry(depth, pixels, torch.tensor([-1.0, 1.0, -0.5, 0.5])).view(12, 16, 4)
  expected_x = (-1.0 + 2.0 * 7.5 / 160) / 4
  expected_y = -(0.5 - 5.5 / 90) / 4
  torch.testing.assert_close(
    points[0, 0], torch.tensor([expected_x, expected_y, 0.25, 1])
  )
  assert points[-1, -1, 0] > 0 and points[-1, -1, 1] > 0
  valid = pixels >= 0
  assert torch.equal(depth[..., 0][valid], raw.flatten()[pixels[valid]])


def test_all_invalid_bins_remain_invalid_and_positive_depth_is_not_clipped():
  raw = torch.full((2, 90, 160, 1), 8.0)
  raw[0] = (
    torch.tensor([0.0, -1.0, float("nan"), float("inf"), -float("inf")])
    .repeat(90 * 160 // 5)
    .reshape(90, 160, 1)
  )
  depth, pixels = reduce_depth(raw)
  points = geometry(depth, pixels, torch.tensor([-1.0, 1.0, -1.0, 1.0])).view(
    2, 12, 16, 4
  )
  assert torch.isfinite(points).all()
  assert not bool(depth[0].any()) and bool((pixels[0] == -1).all())
  assert not bool(points[0].any())
  assert bool((points[1, ..., 2] == 2.0).all())


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
  depth, pixels = reduce_depth(torch.ones(2, 90, 160, 1))
  projected = geometry(depth, pixels, actual)
  assert projected.shape == (2, 768)
  assert torch.equal(projected[0], projected[1]) == shared


def test_thin_near_pixel_survives_both_reductions_and_near_noise_is_not_hidden():
  raw = torch.full((1, 90, 160, 1), 3.0)
  raw[0, 0, 0, 0] = 0.7  # Both old fixed samplers miss this pixel.
  depth, pixels = reduce_depth(raw)
  points = geometry(depth, pixels, torch.tensor([-1.0, 1.0, -1.0, 1.0])).view(12, 16, 4)
  torch.testing.assert_close(points[0, 0, 2:], torch.tensor([0.175, 1.0]))
  # A nearer erroneous positive return also wins: coverage improvement is no denoiser.
  raw[0, 1, 1, 0] = 0.001
  depth, pixels = reduce_depth(raw)
  points = geometry(depth, pixels, torch.tensor([-1.0, 1.0, -1.0, 1.0])).view(12, 16, 4)
  torch.testing.assert_close(points[0, 0, 2], torch.tensor(0.001 / 4))
  assert 161 in pixels[0, :4, :4]


def test_projection_rejects_fabricated_pixel_for_valid_depth():
  depth, pixels = reduce_depth(torch.ones(1, 90, 160, 1))
  pixels[0, 0, 0] = -1
  with pytest.raises(ValueError, match="real original source pixel"):
    geometry(depth, pixels, torch.tensor([-1.0, 1.0, -1.0, 1.0]))
