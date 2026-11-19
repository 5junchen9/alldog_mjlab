"""Wolf八项动作项的首坏子步记录，不套用Black位置动作延迟字段。"""

import torch
from black.fault_capture import write_fault as write_fault


def attach(env, context_provider=lambda: {}):
  import warp as wp
  from support.black.probe import FirstBad, bind

  from wolf.checkpoint import CONTRACT, source_identity

  data = env.sim.wp_data
  fields = tuple(
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
  identity = source_identity()

  def context():
    return {
      **context_provider(),
      "contract": CONTRACT,
      "source_identity": identity,
      "control_tick": env.common_step_counter,
      "episode_length_pre": env.episode_length_buf,
      "requested_action": env.action_manager.action,
      "processed_targets": {
        name: env.action_manager.get_term(name)._processed_actions
        for name in env.action_manager.active_terms
      },
      "field_shapes": {
        name: list(wp.to_torch(getattr(data, name)).shape[1:]) for name in fields
      },
      "replay_limit": "not full contact/solver warm state replay",
    }

  def read():
    return {
      name: wp.to_torch(getattr(data, name)).reshape(env.num_envs, -1)
      for name in fields
    }

  def sync():
    wp.synchronize_device(env.sim.wp_device)
    if torch.device(env.device).type == "cuda":
      torch.cuda.synchronize(env.device)

  probe = FirstBad(read, sync, context)
  bind(env, probe)
  return probe
