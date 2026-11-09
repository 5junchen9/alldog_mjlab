"""Wolf left/right reflection for RSL-RL's native mirror loss.

Observations retain their frame/history layout. The callback returns originals
followed by mirrors, as required by rsl_rl.extensions.Symmetry.compute_loss.
"""

from __future__ import annotations

from typing import cast

import torch
from tensordict import TensorDict

from mjlab.asset_zoo.robots.wolf.wolf_constants import (
  WOLF_JOINTS_PER_LEG,
  WOLF_LEG_ORDER,
)

_OTHER_SIDE = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
_LEG_PERMUTATION = tuple(
  WOLF_LEG_ORDER.index(_OTHER_SIDE[leg]) for leg in WOLF_LEG_ORDER
)
_NUM_LEGS = len(WOLF_LEG_ORDER)
_NUM_LEG_JOINTS = len(WOLF_JOINTS_PER_LEG) - 1
_ACTION_SIGNS = tuple(
  1.0 if joint == "foot" else -1.0
  for _ in WOLF_LEG_ORDER
  for joint in WOLF_JOINTS_PER_LEG
)
_FRAME_TERM_DIMS = (
  3,  # command
  3,  # angular velocity
  3,  # projected gravity
  _NUM_LEGS * _NUM_LEG_JOINTS,
  _NUM_LEGS * _NUM_LEG_JOINTS,
  _NUM_LEGS,  # signed wheel velocities
  len(_ACTION_SIGNS),
)
_FRAME_DIM = sum(_FRAME_TERM_DIMS)


def _swap_legs(values: torch.Tensor, joints_per_leg: int) -> torch.Tensor:
  grouped = values.unflatten(-1, (_NUM_LEGS, joints_per_leg))
  return grouped[..., _LEG_PERMUTATION, :].flatten(-2)


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
  """Reflect raw actions; signed forward wheel targets only swap sides."""
  swapped = _swap_legs(actions, len(WOLF_JOINTS_PER_LEG))
  return swapped * actions.new_tensor(_ACTION_SIGNS)


def mirror_actor_frame(frame: torch.Tensor) -> torch.Tensor:
  """Reflect one frame or every frame of a history without reversing time."""
  command, angular, gravity, pos, vel, wheels, actions = frame.split(
    _FRAME_TERM_DIMS, dim=-1
  )
  return torch.cat(
    (
      command * command.new_tensor((1.0, -1.0, -1.0)),
      angular * angular.new_tensor((-1.0, 1.0, -1.0)),
      gravity * gravity.new_tensor((1.0, -1.0, 1.0)),
      -_swap_legs(pos, _NUM_LEG_JOINTS),
      -_swap_legs(vel, _NUM_LEG_JOINTS),
      _swap_legs(wheels, 1),
      mirror_actions(actions),
    ),
    dim=-1,
  )


def left_right_mirror(
  env,
  obs: TensorDict | None,
  actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """RSL-RL callback for actor, critic and estimator-velocity observations."""
  del env
  augmented_obs: TensorDict | None = None
  if obs is not None:
    mirrored = obs.clone()
    for name, value in obs.items():
      value = cast(torch.Tensor, value)
      if name == "actor":
        mirrored[name] = mirror_actor_frame(value)
      elif name == "critic":
        mirrored[name] = torch.cat(
          (
            mirror_actor_frame(value[..., :_FRAME_DIM]),
            value[..., _FRAME_DIM:] * value.new_tensor((1.0, -1.0, 1.0)),
          ),
          dim=-1,
        )
      elif name == "estimator_velocity":
        mirrored[name] = value * value.new_tensor((1.0, -1.0, 1.0))
      else:
        raise ValueError(f"Unsupported Wolf symmetry observation group: {name}")
    # TensorDict dispatch preserves its container type (the stub also lists Tensor).
    augmented_obs = cast(TensorDict, TensorDict.cat((obs, mirrored), dim=0))
  augmented_actions = None
  if actions is not None:
    augmented_actions = torch.cat((actions, mirror_actions(actions)), dim=0)
  return augmented_obs, augmented_actions
