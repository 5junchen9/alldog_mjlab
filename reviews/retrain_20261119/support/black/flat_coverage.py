"""Retain v28 mixed coverage and fill ordinary pure-axis ramp exposure."""

from support.black.axis_modes import AxisEvents
from support.black.context_coverage import CONTEXT_FIELDS as CONTEXT_FIELDS
from support.black.context_coverage import CONTEXT_POLICY as CONTEXT_POLICY
from support.black.context_coverage import POLICY as POLICY
from support.black.context_coverage import TARGETS as TARGETS
from support.black.context_coverage import FlatCoverage as V28Coverage
from support.black.context_coverage import context_batch as context_batch
from support.black.context_coverage import (
  initialize_context as initialize_context,
)
from support.black.context_coverage import reset_context as reset_context
from support.black.context_coverage import target_masks as target_masks
from support.black.context_coverage import unknown_context as unknown_context


class FlatCoverage(V28Coverage):
  def __init__(self):
    super().__init__()
    self.axis_events = AxisEvents()

  def resample(self, command, ids, before, step):
    super().resample(command, ids, before, step)
    self.axis_events.apply(command, ids, before, step)

  def state_dict(self):
    result = super().state_dict()
    result["axis_modes_v29"] = self.axis_events.state_dict()
    return result

  @classmethod
  def from_state(cls, saved):
    legacy = dict(saved)
    axis = legacy.pop("axis_modes_v29", None)
    old = V28Coverage.from_state(legacy)
    obj = cls()
    obj.__dict__.update(old.__dict__)
    obj.axis_events = AxisEvents.from_state(axis)
    return obj
