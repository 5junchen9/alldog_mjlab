"""First-round candidate experiments; all original task registrations stay intact."""

from copy import deepcopy
from dataclasses import fields

from mjlab.managers.reward_manager import RewardTermCfg

from .candidate_command import CandidateVelocityCommandCfg
from .candidate_rewards import (
  configure_low_rewards,
  linear_velocity_huber_cost,
  yaw_velocity_huber_cost,
)
from .env_cfgs import black_teacher_omni_env_cfg


def black_candidate_env_cfg(profile: str, play: bool = False, *, rewards=True):
  cfg = black_teacher_omni_env_cfg(play=play)
  original = cfg.commands["twist"]
  cfg.commands["twist"] = CandidateVelocityCommandCfg(
    **{f.name: deepcopy(getattr(original, f.name)) for f in fields(original)},
    profile=profile,
  )
  if profile == "low":
    cfg.commands["twist"].ranges = CandidateVelocityCommandCfg.Ranges(
      (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)
    )
    if rewards:
      configure_low_rewards(cfg)
  elif profile == "high":
    if rewards:
      cfg.rewards["linear_huber"] = RewardTermCfg(
        func=linear_velocity_huber_cost,
        weight=-0.20,
        params={"command_name": "twist", "delta": 0.5},
      )
      cfg.rewards["yaw_huber"] = RewardTermCfg(
        func=yaw_velocity_huber_cost,
        weight=-0.10,
        params={"command_name": "twist", "delta": 0.25},
      )
  else:
    raise ValueError("profile must be low or high")
  return cfg
