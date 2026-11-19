"""仿真候选相机扰动；标称安装位置和内参保持可核对。"""

import math

import torch
import warp as wp

from mjlab.managers import EventTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

SENSOR_NOISE = {
  "range_m": [0.2, 4.0],
  "relative_depth_std": 0.015,
  "dropout_probability": 0.01,
  "near_spike_probability": 0.001,
  "mount_position_uniform_m": 0.002,
  "mount_rpy_uniform_degrees": 0.5,
  "calibration_status": "simulation_candidate_not_hardware_measurement",
}


@requires_model_fields("cam_pos", "cam_quat")
def randomize_mount(env, env_ids):
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  model = env.sim.mj_model
  camera = model.camera("shared_actor_depth").id
  pos = torch.as_tensor(model.cam_pos[camera].copy(), device=env.device)
  quat = torch.as_tensor(model.cam_quat[camera].copy(), device=env.device)
  shift = torch.empty(len(ids), 3, device=env.device).uniform_(-0.002, 0.002)
  angles = torch.empty(len(ids), 3, device=env.device).uniform_(
    -math.radians(0.5), math.radians(0.5)
  )
  delta = quat_from_euler_xyz(*angles.unbind(-1))
  wp.to_torch(env.sim.wp_model.cam_pos)[ids, camera] = (pos + shift).float()
  wp.to_torch(env.sim.wp_model.cam_quat)[ids, camera] = quat_mul(
    quat.expand(len(ids), -1).float(), delta
  )


def configure(cfg):
  cfg.events["camera_mount"] = EventTermCfg(func=randomize_mount, mode="reset")
