"""Single-policy command mixture with self-contained boundary recovery cycles.

Low/Omni/high own fixed 35/35/30 percent environment partitions across resets.
Recovery holds longer than normal events; both event and held-time shares are logged.
Phases: -1 normal, 0 initialization settle, 1 boundary, 2 reverse, 3 stop.
A phase change is not a new bucket event. No teacher or policy history is used.
"""

from dataclasses import dataclass

import torch

from .omni_command import OmniVelocityCommand, OmniVelocityCommandCfg

PROFILE_NAMES = ("low", "omni", "high")
PROFILE_WEIGHTS = (0.35, 0.35, 0.30)
BUCKET_NAMES = (
  "low_pure",
  "low_mixed",
  "omni_pure",
  "omni_mixed",
  "high_pure",
  "boundary_recovery",
  "stand",
)
RECOVERY_BUCKET = 5
STAND_BUCKET = 6
OMNI_WEIGHTS = (0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.15, 0.05)
PHASE_NAMES = ("settle", "boundary", "reverse", "stop")
PHASE_SECONDS = (1.0, 5.0, 5.0, 3.0)


def allocate_profiles(count: int, device: str | torch.device, seed: int = 0):
  """Largest-remainder counts, then a reproducible shuffled permanent partition.

  For 4096 environments this is 1434 low, 1433 Omni, 1229 high. At very small
  counts, largest-remainder allocation still sums exactly to the environment count.
  """
  raw = [count * weight for weight in PROFILE_WEIGHTS]
  counts = [int(value) for value in raw]
  order = sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))
  for i in order[: count - sum(counts)]:
    counts[i] += 1
  profiles = torch.repeat_interleave(torch.arange(3), torch.tensor(counts))
  generator = torch.Generator(device="cpu").manual_seed(seed)
  return profiles[torch.randperm(count, generator=generator)].to(device)


def sample_targets(profile_ids: torch.Tensor, ranges=None):
  """Sample within each fixed profile. Returns targets and stable bucket IDs.

  Low: 70% pure / 20% mixed / 10% stand. Omni: original eight categories
  and uniform ranges, including the original pure-axis minimum of 0.1.
  High: 50% pure / 50% complete boundary recovery cycle.
  """
  count, device = len(profile_ids), profile_ids.device
  target = torch.zeros(count, 3, device=device)
  bucket = torch.full((count,), STAND_BUCKET, device=device, dtype=torch.long)
  draw = torch.rand(count, device=device)
  low, omni, high = profile_ids == 0, profile_ids == 1, profile_ids == 2
  bucket[low & (draw < 0.7)] = 0
  bucket[low & (draw >= 0.7) & (draw < 0.9)] = 1
  bucket[high & (draw < 0.5)] = 4
  bucket[high & (draw >= 0.5)] = RECOVERY_BUCKET
  direction = torch.randint(6, (count,), device=device)
  axis, sign = direction // 2, 1 - 2 * (direction % 2)
  lower = torch.where(high, torch.where(axis == 2, 0.3, 0.5), 0.05)
  upper = torch.where(high, torch.where(axis == 2, 0.5, 1.0), 0.30)
  target[torch.arange(count, device=device), axis] = (
    lower + torch.rand(count, device=device) * (upper - lower)
  ) * sign
  angle = torch.rand(count, device=device) * (2 * torch.pi)
  speed = 0.1 + 0.4 * torch.rand(count, device=device)
  mixed = torch.stack(
    (
      speed * angle.cos(),
      speed * angle.sin(),
      torch.rand(count, device=device) * 0.6 - 0.3,
    ),
    dim=-1,
  )
  target[bucket == 1] = mixed[bucket == 1]
  signs = torch.randint(2, (count, 3), device=device) * 2 - 1
  boundary = signs * torch.tensor([1.0, 1.0, 0.5], device=device)
  target[bucket == RECOVERY_BUCKET] = boundary[bucket == RECOVERY_BUCKET]
  omni_ids = omni.nonzero().flatten()
  if len(omni_ids):
    category = torch.multinomial(
      torch.tensor(OMNI_WEIGHTS, device=device), len(omni_ids), replacement=True
    )
    limits = (
      ((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5))
      if ranges is None
      else (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
    )
    commands = torch.empty(len(omni_ids), 3, device=device)
    for i, bounds in enumerate(limits):
      commands[:, i].uniform_(*bounds)
    for index, coordinate, polarity in (
      (0, 0, 1),
      (1, 0, -1),
      (2, 2, 1),
      (3, 2, -1),
      (4, 1, 1),
      (5, 1, -1),
    ):
      selected = category == index
      magnitude = commands[selected, coordinate].abs().clamp(min=0.1)
      commands[selected] = 0
      commands[selected, coordinate] = magnitude * polarity
    commands[category == 7] = 0
    target[omni_ids] = commands
    bucket[omni_ids] = torch.where(
      category < 6, 2, torch.where(category == 6, 3, STAND_BUCKET)
    )
  target[bucket == STAND_BUCKET] = 0
  return target, bucket


class TeacherFreeVelocityCommand(OmniVelocityCommand):
  def __init__(self, cfg: "TeacherFreeVelocityCommandCfg", env):
    super().__init__(cfg, env)
    self.teacherfree_cfg = cfg
    self.target_command = torch.zeros_like(self.command)
    self.cycle_target = torch.zeros_like(self.command)
    self.profile_id = allocate_profiles(self.num_envs, self.device, cfg.profile_seed)
    self.task_class = torch.full((self.num_envs,), STAND_BUCKET, device=self.device)
    self.cycle_phase = torch.full_like(self.task_class, -1)
    self.phase_age = torch.zeros(self.num_envs, device=self.device)
    self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.abrupt = torch.ones_like(self.initialized)
    self.sample_counts = torch.zeros(
      len(BUCKET_NAMES), dtype=torch.long, device=self.device
    )
    self.held_seconds = torch.zeros(len(BUCKET_NAMES), device=self.device)
    self.profile_event_counts = torch.zeros(3, dtype=torch.long, device=self.device)
    self.profile_held_seconds = torch.zeros(3, device=self.device)
    self.phase_seconds = torch.zeros(4, device=self.device)
    self.transition_mode_counts = torch.zeros(2, dtype=torch.long, device=self.device)
    self.transition_mode_seconds = torch.zeros(2, device=self.device)
    self._command_dt = torch.zeros(self.num_envs, device=self.device)

  @property
  def current_command(self):
    return self.command

  @property
  def transition_mode(self):
    """-1 outside recovery; 0 ramp, 1 abrupt, fixed for the entire cycle."""
    return torch.where(self.task_class == RECOVERY_BUCKET, self.abrupt.long(), -1)

  def reset(self, env_ids):
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    elif isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
    self.initialized[env_ids] = False
    self.cycle_phase[env_ids] = -1
    self.phase_age[env_ids] = 0
    self.cycle_target[env_ids] = 0
    self.target_command[env_ids] = 0
    self.vel_command_b[env_ids] = 0
    self._command_dt[env_ids] = 0
    return super().reset(env_ids)

  def _resample_command(self, env_ids):
    # The manager calls this on timer expiry, so advance an existing cycle
    # before sampling a new event. The cycle target is never overwritten early.
    active = (self.cycle_phase[env_ids] >= 0) & (self.cycle_phase[env_ids] < 3)
    cycle_ids = env_ids[active]
    if len(cycle_ids):
      self.cycle_phase[cycle_ids] += 1
      self._set_phase(cycle_ids)
    new_ids = env_ids[~active]
    if len(new_ids):
      target, bucket = sample_targets(self.profile_id[new_ids], self.cfg.ranges)
      self.task_class[new_ids] = bucket
      self.cycle_phase[new_ids] = -1
      self.cycle_target[new_ids] = 0
      self.target_command[new_ids] = target
      self.phase_age[new_ids] = 0
      self.abrupt[new_ids] = True
      self.sample_counts += torch.bincount(bucket, minlength=len(BUCKET_NAMES))
      self.profile_event_counts += torch.bincount(self.profile_id[new_ids], minlength=3)
      self.time_left[new_ids] = torch.empty(len(new_ids), device=self.device).uniform_(
        *self.cfg.resampling_time_range
      )
      recovery_ids = new_ids[bucket == RECOVERY_BUCKET]
      if len(recovery_ids):
        self.cycle_target[recovery_ids] = target[bucket == RECOVERY_BUCKET]
        self.abrupt[recovery_ids] = (
          torch.rand(len(recovery_ids), device=self.device)
          < self.teacherfree_cfg.abrupt_probability
        )
        self.transition_mode_counts += torch.bincount(
          self.abrupt[recovery_ids].long(), minlength=2
        )
        # Settling is only for a newly reset robot, not every cycle entrance.
        self.cycle_phase[recovery_ids] = self.initialized[recovery_ids].long()
        self._set_phase(recovery_ids)
      self.initialized[new_ids] = True
    self.is_standing_env[env_ids] = False
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False
    immediate = self.abrupt[env_ids] | (self.cycle_phase[env_ids] == 0)
    immediate_ids = env_ids[immediate]
    self.vel_command_b[immediate_ids] = self.target_command[immediate_ids]

  def _set_phase(self, env_ids):
    phase = self.cycle_phase[env_ids]
    sign = torch.where(phase == 1, 1, torch.where(phase == 2, -1, 0))
    self.target_command[env_ids] = self.cycle_target[env_ids] * sign[:, None]
    self.time_left[env_ids] = torch.tensor(PHASE_SECONDS, device=self.device)[phase]
    self.phase_age[env_ids] = 0

  def compute(self, dt, env_ids=None):
    """Advance exact elapsed time, including crossing multiple phase boundaries.

    Per-env dt=0 leaves that environment untouched. Split at timer boundaries so
    a ramp never receives elapsed time belonging to the preceding phase.
    """
    if env_ids is not None and isinstance(dt, torch.Tensor):
      raise ValueError("Per-env tensor dt requires env_ids=None")
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    remaining = torch.as_tensor(dt, device=self.device).expand(len(ids)).clone()
    if (remaining < 0).any() or not torch.isfinite(remaining).all():
      raise ValueError("dt must be finite and nonnegative")
    # Preserve Uniform's episode displacement bookkeeping and fresh spawn pose.
    self._integrate_command(dt, env_ids)
    fresh = ids[self._env.episode_length_buf[ids] == 0]
    self.episode_start_pos_w[fresh] = self.robot.data.root_link_pos_w[fresh, :2]
    while len(ids):
      expired = ids[self.time_left[ids] <= 0]
      if len(expired):
        self._resample(expired)
      elapsed = torch.minimum(remaining, self.time_left[ids])
      self._command_dt[ids] = elapsed
      self.held_seconds.scatter_add_(0, self.task_class[ids], elapsed)
      self.profile_held_seconds.scatter_add_(0, self.profile_id[ids], elapsed)
      recovery = self.task_class[ids] == RECOVERY_BUCKET
      self.phase_seconds.scatter_add_(
        0, self.cycle_phase[ids[recovery]], elapsed[recovery]
      )
      self.transition_mode_seconds.scatter_add_(
        0, self.abrupt[ids[recovery]].long(), elapsed[recovery]
      )
      self._update_command(ids)
      self.time_left[ids] -= elapsed
      self.phase_age[ids] += elapsed
      remaining -= elapsed
      # Change target at the endpoint, but do not ramp toward it before its time.
      expired = ids[self.time_left[ids] <= 0]
      if len(expired):
        self._resample(expired)
      pending = remaining > 0
      ids, remaining = ids[pending], remaining[pending]
    self._update_metrics()

  def _update_command(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    current = self.command[ids]
    delta = self.target_command[ids] - current
    dt = self._command_dt[ids]
    factor = (
      dt
      * self.teacherfree_cfg.linear_acceleration
      / delta[:, :2].norm(dim=-1).clamp(min=1e-12)
    ).clamp(max=1)
    delta[:, :2] *= factor[:, None]
    yaw_limit = dt * self.teacherfree_cfg.yaw_acceleration
    delta[:, 2] = delta[:, 2].clamp(-yaw_limit, yaw_limit)
    self.vel_command_b[ids] = torch.where(
      self.abrupt[ids, None], self.target_command[ids], current + delta
    )
    self._resolve_command(self.robot.data.heading_w, env_ids)

  def _update_metrics(self):
    super()._update_metrics()
    log = self._env.extras["log"]
    total_events = self.sample_counts.sum().clamp(min=1)
    total_time = self.held_seconds.sum().clamp(min=1e-12)
    for i, name in enumerate(BUCKET_NAMES):
      log[f"TeacherFreeEvents/{name}"] = self.sample_counts[i].clone()
      log[f"TeacherFreeEventShare/{name}"] = self.sample_counts[i] / total_events
      log[f"TeacherFreeHeldShare/{name}"] = self.held_seconds[i] / total_time
      log[f"TeacherFreeBuckets/{name}"] = (self.task_class == i).float().mean()
    for i, name in enumerate(PROFILE_NAMES):
      log[f"TeacherFreeProfile/{name}"] = (self.profile_id == i).float().mean()
      log[f"TeacherFreeProfileEvents/{name}"] = self.profile_event_counts[i].clone()
      log[f"TeacherFreeProfileHeldShare/{name}"] = (
        self.profile_held_seconds[i] / total_time
      )
    for i, name in enumerate(PHASE_NAMES):
      log[f"TeacherFreePhase/{name}"] = (self.cycle_phase == i).float().mean()
      log[f"TeacherFreePhaseSeconds/{name}"] = self.phase_seconds[i].clone()
    for i, name in enumerate(("ramp", "abrupt")):
      log[f"TeacherFreeModeEvents/{name}"] = self.transition_mode_counts[i].clone()
      log[f"TeacherFreeModeSeconds/{name}"] = self.transition_mode_seconds[i].clone()
      log[f"TeacherFreeMode/{name}"] = (self.transition_mode == i).float().mean()


@dataclass(kw_only=True)
class TeacherFreeVelocityCommandCfg(OmniVelocityCommandCfg):
  profile_seed: int = 0
  abrupt_probability: float = 0.5
  linear_acceleration: float = 1.0
  yaw_acceleration: float = 0.5

  def build(self, env):
    if not 0 <= self.abrupt_probability <= 1:
      raise ValueError("abrupt_probability must be in [0, 1]")
    if self.linear_acceleration <= 0 or self.yaw_acceleration <= 0:
      raise ValueError("accelerations must be positive")
    if not 0 < self.resampling_time_range[0] <= self.resampling_time_range[1]:
      raise ValueError("resampling times must be positive and ordered")
    return TeacherFreeVelocityCommand(self, env)
