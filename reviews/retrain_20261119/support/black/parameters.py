"""Simulation-only starting bounds; dynamic diagnostics substantiate them."""

PARAMETERS = dict(
  stage_credits=[0.25, 0.75, 2.0],
  settle_steps=6,
  maximum_step_progress=0.25,
  minimum_trials=32,
  window=64,
  success_fraction=0.90,
  finish_support_z_tolerance=0.08,
  finish_max_linear_speed=0.12,
  finish_max_angular_speed=0.7,
  finish_max_tilt_radians=0.35,
)
RATIONALE = dict(
  scope="simulation; no hardware certification claimed",
  settle="6 x 20ms control steps on exit apron with four terrain contacts",
  event="0.25 once after safe ordered full-foot exit;2.75 only after unchanged strict stable finish; episode cap3; failed episodes receive no terminal credit",
  displacement="0.25m/control step rejects resets/teleports above requested 0.35m/s",
  safety="retain original thigh/trunk/tilt/clearance termination and soft joint/collision rewards",
  curriculum="each recorded height; 32 trials per entry angle and direction; last64 success >=90%; earlier20% replay",
)
