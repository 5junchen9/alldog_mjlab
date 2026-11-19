"""首坏子步保留更多物理状态；诊断记录仍不冒充完整求解器重放。"""

import importlib.metadata
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def attach(env, context_provider=lambda: {}):
  import warp as wp
  from support.black.probe import FirstBad, bind

  from black.checkpoint import CONTRACT
  from black.eval_collect import source_files
  from black.eval_protocol import digest

  data = env.sim.wp_data
  names = tuple(
    name
    for name in (
      "qpos",
      "qvel",
      "ctrl",
      "qacc",
      "qacc_warmstart",
      "act",
      "qfrc_applied",
      "xfrc_applied",
      "qfrc_actuator",
    )
    if hasattr(data, name)
    and wp.to_torch(getattr(data, name)).ndim >= 2
    and wp.to_torch(getattr(data, name)).shape[0] == env.num_envs
  )
  versions = {
    name: importlib.metadata.version(name)
    for name in ("torch", "mujoco", "mujoco-warp", "warp-lang")
  }
  action = env.action_manager.get_term("joint_pos")
  source_sha256 = digest(source_files())

  def context():
    return dict(
      context_provider(),
      control_tick=env.common_step_counter,
      physics_step_before_control=env._sim_step_counter,
      episode_length_pre=env.episode_length_buf,
      action_delay=action.control_delay.lags,
      action_history=action.control_delay.history,
      processed_target=action._processed_actions,
      contract=CONTRACT,
      versions=versions,
      source_sha256=source_sha256,
      captured_physics_fields=names,
      physics_field_shapes={
        name: tuple(wp.to_torch(getattr(data, name)).shape[1:]) for name in names
      },
      device=str(env.device),
      gpu_name=(
        torch.cuda.get_device_name(env.device)
        if torch.device(env.device).type == "cuda"
        else None
      ),
      torch_cpu_rng=torch.get_rng_state(),
      torch_cuda_rng=(
        torch.cuda.get_rng_state(env.device)
        if torch.device(env.device).type == "cuda"
        else None
      ),
      numpy_rng=repr(np.random.get_state()),
      python_rng=repr(random.getstate()),
      replay_limit="contact/solver internal state and pre-event model are not restored",
    )

  def sync():
    wp.synchronize_device(env.sim.wp_device)
    if torch.device(env.device).type == "cuda":
      torch.cuda.synchronize(env.device)

  def read():
    # FirstBad按world检查；原始字段shape另列，展平不丢数据。
    return {
      name: wp.to_torch(getattr(data, name)).reshape(env.num_envs, -1) for name in names
    }

  probe = FirstBad(read, sync, context)
  bind(env, probe)
  return probe


def write_fault(probe, directory):
  """完整写入后发布首坏文件；磁盘失败不能留下看似有效的正式半包。"""
  if probe.fault is None:
    raise ValueError("no captured fault")
  directory = Path(directory)
  directory.mkdir(exist_ok=False)
  descriptor, name = tempfile.mkstemp(dir=directory, suffix=".pending")
  temporary = Path(name)
  target = directory / "FIRST_BAD.pt"
  try:
    with os.fdopen(descriptor, "wb") as stream:
      torch.save(probe.fault, stream)
      stream.flush()
      os.fsync(stream.fileno())
    os.link(temporary, target)
  finally:
    temporary.unlink(missing_ok=True)
  descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)
  return target
