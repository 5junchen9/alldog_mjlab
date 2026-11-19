"""Select the opt-in workspace scheduler in this isolated process only.

Scratch rows bound concurrent workers, not the number of collision candidates.
Each worker completes its candidate before reusing its row. All original contact
capacity checks remain in the Apache-2.0 vendored kernels.
"""

from mujoco_warp._src import collision_driver
from mujoco_warp._src.collision_convex import convex_narrowphase as native_narrowphase

from mjlab.sim._vendor import collision_convex_workspace

_UNSET = object()
_selected = _UNSET


def select_collision_workspace(slots: int | None) -> None:
  global _selected
  if _selected is not _UNSET and _selected != slots:
    raise RuntimeError(
      "Native/optimized collision modes require separate fresh processes"
    )
  _selected = slots
  if slots is None:
    collision_driver.convex_narrowphase = native_narrowphase
  else:
    collision_convex_workspace.SCRATCH_SLOTS = slots
    collision_driver.convex_narrowphase = collision_convex_workspace.convex_narrowphase
