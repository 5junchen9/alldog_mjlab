"""Teacher-free single-actor continuation; preserve original Omni training setup."""

from copy import deepcopy
from dataclasses import fields

from .env_cfgs import black_teacher_omni_env_cfg
from .teacherfree_command import TeacherFreeVelocityCommandCfg


def black_teacherfree_env_cfg(play: bool = False):
  cfg = black_teacher_omni_env_cfg(play=play)
  original = cfg.commands["twist"]
  kwargs = {f.name: deepcopy(getattr(original, f.name)) for f in fields(original)}
  kwargs["resampling_time_range"] = (3.0, 8.0)
  cfg.commands["twist"] = TeacherFreeVelocityCommandCfg(**kwargs)
  cfg.auto_reset = True
  return cfg
