"""Synchronous first-fault diagnosis; never repairs or advances physics."""

from pathlib import Path

import torch


class InvalidPhysics(FloatingPointError):
  pass


def owned(value, lanes=None, worlds=None):
  """Freeze context; on publication retain only the selected worlds."""
  if isinstance(value, torch.Tensor):
    if lanes is not None and value.ndim and value.shape[0] == worlds:
      value = value[lanes.to(value.device)]
    value = value.detach().clone()
    return value.cpu() if lanes is not None else value
  if isinstance(value, dict):
    return {key: owned(item, lanes, worlds) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [owned(item, lanes, worlds) for item in value]
  return value


class FirstBad:
  """Observe operation boundaries and retain exactly one bounded receipt."""

  def __init__(self, read, sync, context_provider=lambda: {}):
    self.read = read
    self.sync = sync
    self.context_provider = context_provider
    self.context = None
    self.substep = 0
    self.fault = None
    self.reset_ids = None
    self.pending_reset_write = False
    self._originals = []

  def snapshot(self):
    self.sync()
    return {key: value.detach().clone() for key, value in self.read().items()}

  @staticmethod
  def bad(data):
    return torch.stack([~torch.isfinite(value).all(1) for value in data.values()]).any(
      0
    )

  def record(self, operation, boundary, pre, post, error):
    """Recording must never replace the exception that stopped the operation."""
    if self.fault is not None:
      return
    self.fault = dict(
      stage=f"{operation}:{boundary}",
      substep_1based=self.substep if operation == "sim.step" else None,
      last_physics_substep_1based=self.substep,
      error=repr(error),
      replayable=False,
      scope="native operation boundary; not a solver iteration or full replay",
      bad_lane_count=None,
    )
    try:
      if pre is None or post is None:
        # A failed device synchronization cannot establish which world failed.
        return
      bad = self.bad(pre) | self.bad(post)
      all_lanes = bad.nonzero().flatten()
      lanes = all_lanes[:8]
      worlds = len(bad)
      self.fault.update(
        bad_lane_count=int(bad.sum()),
        bad_lanes=all_lanes.cpu().clone(),
        detailed_lanes=lanes.cpu().clone(),
        pre={key: value[lanes].cpu().clone() for key, value in pre.items()},
        post={key: value[lanes].cpu().clone() for key, value in post.items()},
        context=owned(self.context, lanes, worlds),
        reset_env_ids=None if self.reset_ids is None else self.reset_ids.cpu().clone(),
      )
    except Exception as capture_error:
      self.fault["capture_error"] = repr(capture_error)

  def boundary(self, operation, original, *args, **kwargs):
    if self.fault is not None:
      raise RuntimeError("probe has already stopped on its first fault")
    standalone = self.context is None
    if standalone:
      self.context = owned(self.context_provider())
      self.substep = 0
    pre = None
    try:
      try:
        pre = self.snapshot()
      except BaseException as error:
        self.record(operation, "before_snapshot_failed", None, None, error)
        raise
      if bool(self.bad(pre).any()):
        error = InvalidPhysics(f"nonfinite native input before {operation}")
        self.record(operation, "before", pre, pre, error)
        raise error
      try:
        result = original(*args, **kwargs)
      except BaseException as error:
        try:
          post = self.snapshot()
        except BaseException as capture_error:
          self.record(operation, "raised", pre, None, error)
          self.fault["capture_error"] = repr(capture_error)
        else:
          self.record(operation, "raised", pre, post, error)
        raise
      try:
        post = self.snapshot()
      except BaseException as error:
        self.record(operation, "after_snapshot_failed", pre, None, error)
        raise
      if bool(self.bad(post).any()):
        error = InvalidPhysics(f"nonfinite native state after {operation}")
        self.record(operation, "after", pre, post, error)
        raise error
      return result
    finally:
      if standalone:
        self.context = None

  def step(self, original):
    standalone = self.context is None
    if standalone:
      self.context = owned(self.context_provider())
      self.substep = 0
    self.substep += 1
    try:
      return self.boundary("sim.step", original)
    finally:
      if standalone:
        self.context = None

  def control(self, original, action):
    if self.context is not None or self.fault is not None:
      raise RuntimeError("nested control or an already faulted probe")
    self.context = owned(self.context_provider())
    self.context["requested_action"] = action.detach().clone()
    self.substep = 0
    self.reset_ids = None
    try:
      return original(action)
    finally:
      self.context = None

  def detach(self):
    """Restore only this instance's methods; no reset or extra simulation call."""
    for obj, name, had_instance_value, value in reversed(self._originals):
      if had_instance_value:
        setattr(obj, name, value)
      else:
        delattr(obj, name)
    self._originals.clear()


def attach(env, context_provider=lambda: {}):
  """Attach after env construction; caller authenticates the actual source.

  The provider supplies round, unique episode, command and source identity.
  Tensor values with a first dimension of num_envs are sliced to fault worlds.
  This function neither verifies Windows provenance nor changes CUDA graphs.
  """
  import warp as wp

  def context():
    return dict(
      context_provider(),
      control_tick=env.common_step_counter,
      physics_step_before_control=env._sim_step_counter,
      episode_length_pre=env.episode_length_buf,
    )

  def sync():
    wp.synchronize_device(env.sim.wp_device)
    if torch.device(env.device).type == "cuda":
      torch.cuda.synchronize(env.device)

  def read():
    return {
      name: wp.to_torch(getattr(env.sim.wp_data, name))
      for name in ("qpos", "qvel", "ctrl")
    }

  probe = FirstBad(read, sync, context)
  bind(env, probe)
  return probe


def bind(env, probe):
  """Instance hooks, also exercised with a CPU fake environment in tests."""
  targets = [
    (env, "step"),
    (env, "_reset_idx"),
    (env.sim, "step"),
    (env.sim, "reset"),
    (env.sim, "forward"),
    (env.sim, "sense"),
    (env.event_manager, "apply"),
    (env.scene, "write_data_to_sim"),
  ]
  if any(name in vars(obj) for obj, name in targets):
    raise ValueError("diagnostic requires unwrapped instance methods")

  def replace(obj, name, make_wrapper):
    original = getattr(obj, name)
    probe._originals.append((obj, name, name in vars(obj), vars(obj).get(name)))
    setattr(obj, name, make_wrapper(original))

  replace(env, "step", lambda original: lambda action: probe.control(original, action))
  replace(env.sim, "step", lambda original: lambda: probe.step(original))
  for name in ("reset", "forward", "sense"):
    replace(
      env.sim,
      name,
      lambda original, name=name: (
        lambda *args, **kwargs: probe.boundary(f"sim.{name}", original, *args, **kwargs)
      ),
    )

  def event_wrapper(original):
    def apply(mode, *args, **kwargs):
      return probe.boundary(f"event.{mode}", original, mode, *args, **kwargs)

    return apply

  def reset_wrapper(original):
    def reset(env_ids=None):
      probe.reset_ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else env_ids.detach().clone()
      )
      probe.pending_reset_write = True
      return probe.boundary("env._reset_idx", original, env_ids)

    return reset

  def write_wrapper(original):
    def write():
      if not probe.pending_reset_write:
        return probe.boundary("scene.action_write", original)
      result = probe.boundary("scene.reset_write", original)
      probe.pending_reset_write = False
      return result

    return write

  replace(env.event_manager, "apply", event_wrapper)
  replace(env, "_reset_idx", reset_wrapper)
  replace(env.scene, "write_data_to_sim", write_wrapper)


def write_fault(probe, directory):
  """Call on the outer error path, then re-raise its original exception."""
  if probe.fault is None:
    raise ValueError("no captured fault")
  directory = Path(directory)
  directory.mkdir(exist_ok=False)
  with (directory / "FIRST_BAD.pt").open("xb") as stream:
    torch.save(probe.fault, stream)
  return directory / "FIRST_BAD.pt"
