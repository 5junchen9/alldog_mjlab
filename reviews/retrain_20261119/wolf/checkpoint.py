"""Wolf独立输入合同、原子存档、严格同源恢复；不接受Black或旧Wolf权重。"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from black.checkpoint import tensor_digest

from wolf.camera import CAMERA
from wolf.model import initial_models

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = {
  "schema": "wolf_shared_actor_v1_scratch",
  "history_shape": [6, 53],
  "input_order": [
    "current53",
    "HIM_next_velocity3_scaled2",
    "HIM_unit16",
    "visible_geometry32",
  ],
  "action_order": [
    f"{leg}_{joint}"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf", "foot")
  ],
  "leg_action_scales": [0.125, 0.25, 0.25],
  "wheel_action_scale": 5.0,
  "wheel_signs": [1, -1, 1, -1],
  "camera": CAMERA,
  "depth": "planar_m_nearest_valid_surface160x90_to64x48",
  "geometry": "same_depth_original_ray_xyz_div4_valid_12x16",
  "control_hz": 50,
  "camera_hz": 10,
  "critic": "unnoised_proprio53_true_velocity3_terrain187_valid187",
}


def source_identity():
  from black.eval_collect import runtime_identity, source_files

  files = source_files()
  # 代码/资产身份跨机稳定；运行库另列，仅完整学习器恢复严格绑定。
  files.pop("@runtime/versions.json", None)
  for directory in ("wolf", "support/wolf"):
    for path in sorted((ROOT / directory).rglob("*")):
      if path.is_file() and path.suffix in (".py", ".json"):
        files[str(path.relative_to(ROOT))] = hashlib.sha256(
          path.read_bytes()
        ).hexdigest()
  files = {
    name: value
    for name, value in files.items()
    if not {"tests", "docs"}.intersection(Path(name).parts)
    and not Path(name).name.startswith("test_")
  }
  encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
  return {"sha256": hashlib.sha256(encoded).hexdigest(), "runtime": runtime_identity()}


def teacher_identity(teacher):
  return hashlib.sha256(
    (
      tensor_digest(teacher.state_dict()) + json.dumps(CONTRACT, sort_keys=True)
    ).encode()
  ).hexdigest()


def save_bundle(path, teacher, cnn, gru, *, phase, updates, learning_state=None):
  packet = {
    "contract": CONTRACT,
    "teacher_id": teacher_identity(teacher),
    "teacher": teacher.state_dict(),
    "cnn": cnn.state_dict(),
    "gru": gru.state_dict(),
    "initialization": teacher.initialization,
    "phase": phase,
    "completed_updates": updates,
    "learning_state": learning_state,
    "qualification": "not_evaluated",
    "source_identity": source_identity(),
    "exact_physical_resume": False,
  }
  path = Path(path)
  descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".pending")
  temporary = Path(name)
  try:
    with os.fdopen(descriptor, "wb") as stream:
      torch.save(packet, stream)
      stream.flush()
      os.fsync(stream.fileno())
    os.link(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)
  return packet["teacher_id"]


def load_bundle(path, device="cpu"):
  packet = torch.load(path, map_location=device, weights_only=True)
  if packet["contract"] != CONTRACT:
    raise ValueError("Wolf共享actor输入或动作合同不一致")
  if packet["source_identity"]["sha256"] != source_identity()["sha256"]:
    raise ValueError("源码或资产改变，拒绝载入")
  teacher, cnn, gru = initial_models(packet["initialization"]["seed"])
  teacher.initialization = packet["initialization"]
  for module, key in ((teacher, "teacher"), (cnn, "cnn"), (gru, "gru")):
    module.load_state_dict(packet[key], strict=True)
    module.to(device)
  if teacher_identity(teacher) != packet["teacher_id"]:
    raise ValueError("Wolf teacher权重身份不一致")
  return teacher, cnn, gru, packet
