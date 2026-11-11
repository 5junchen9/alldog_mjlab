"""Continuation preserves Omni rewards, 72D observations and robot configuration."""

from copy import deepcopy
from dataclasses import fields

from .continuation_command import STAGE_WEIGHTS, ContinuationVelocityCommandCfg
from .env_cfgs import black_teacher_omni_env_cfg


def black_continuation_env_cfg(stage: str = "stage1", play: bool = False):
  if stage not in STAGE_WEIGHTS:
    raise ValueError("stage must be stage1 or stage2")
  cfg = black_teacher_omni_env_cfg(play=play)
  original = cfg.commands["twist"]
  cfg.commands["twist"] = ContinuationVelocityCommandCfg(
    **{f.name: deepcopy(getattr(original, f.name)) for f in fields(original)},
    stage=stage,
  )
  cfg.auto_reset = True
  return cfg
