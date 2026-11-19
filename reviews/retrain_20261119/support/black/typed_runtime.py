"""Validated structural runtime interfaces for the generated-terrain branch."""

from collections.abc import Callable
from typing import Protocol, cast

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict


class TerrainState(Protocol):
  terrain_levels: torch.Tensor
  terrain_types: torch.Tensor
  terrain_origins: torch.Tensor
  env_origins: torch.Tensor


def require_terrain(env) -> TerrainState:
  terrain = env.scene.terrain
  for name in ("terrain_levels", "terrain_types", "terrain_origins", "env_origins"):
    if not isinstance(getattr(terrain, name, None), torch.Tensor):
      raise ValueError("Generated terrain tensor missing: " + name)
  return cast(TerrainState, terrain)


class FlatCommand(Protocol):
  flat_mask: torch.Tensor


def require_flat_command(env) -> FlatCommand:
  command = env.command_manager.get_term("twist")
  if not isinstance(getattr(command, "flat_mask", None), torch.Tensor):
    raise ValueError("Explicit flat-domain tensor missing")
  return cast(FlatCommand, command)


def install_hooks(
  algorithm: PPO,
  *,
  process: Callable[
    [TensorDict, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], None
  ]
  | None = None,
  update: Callable[[], dict] | None = None,
) -> None:
  """Bind typed per-instance instrumentation, preserving the framework API."""
  if process is not None:
    algorithm.__dict__.update(process_env_step=process)
  if update is not None:
    algorithm.__dict__.update(update=update)


@torch.inference_mode()
def reset_in_inference(env, env_ids):
  """Manual reset shares the inference context that allocated sensor buffers."""
  return env.reset(env_ids=env_ids)
