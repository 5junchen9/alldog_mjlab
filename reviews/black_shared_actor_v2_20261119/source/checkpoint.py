"""v2随机初始化与自包含学习器；明确拒绝v1权重身份。"""

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from model import Teacher, make_modules

CONTRACT = {
  "schema": "black_shared_actor_v2",
  "input_order": ["current_p45", "HIM_next_velocity3", "HIM_unit16", "terrain32"],
  "history_shape": [5, 45],
  "geometry": "nearest_surface_pool_original_pixel_optical_xyz_div4_valid_12x16_v2",
  "depth": "planar_z_metres_nearest_valid_surface_bins160x90_to64x48_v2",
  "camera_hz": 10,
  "control_hz": 50,
  "action_order": [
    f"{leg}_{joint}"
    for leg in ("FL", "FR", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
  ],
  "action_scale": 0.25,
  "action_space": "policy_mean_before_native_delay_and_PD",
}


def tensor_digest(values):
  result = hashlib.sha256()
  for name, value in sorted(values.items()):
    value = value.detach().cpu().contiguous()
    result.update(str((name, tuple(value.shape), str(value.dtype))).encode())
    result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
  return result.hexdigest()


def teacher_identity(teacher):
  weights = tensor_digest(teacher.state_dict())
  return hashlib.sha256(
    (weights + json.dumps(CONTRACT, sort_keys=True)).encode()
  ).hexdigest()


def initial_models(device="cpu"):
  """调用前设置随机种子；不打开任何历史模型文件。"""
  him, actor, cnn, gru = make_modules()
  teacher = Teacher(him, actor).to(device)
  return teacher, cnn.to(device), gru.to(device)


def save_bundle(path, teacher, cnn, gru, *, phase, updates, learning_state=None):
  """保存新候选身份；没有能力验收结果就不标成合格教师。"""
  packet = {
    "contract": CONTRACT,
    "teacher_id": teacher_identity(teacher),
    "teacher": teacher.state_dict(),
    "cnn": cnn.state_dict(),
    "gru": gru.state_dict(),
    "initialization": copy.deepcopy(teacher.initialization),
    "phase": phase,
    "completed_updates": updates,
    "qualification": "not_evaluated",
    "learning_state": learning_state,
    "exact_physical_resume": False,
  }
  path = Path(path)
  descriptor, name = tempfile.mkstemp(
    dir=path.parent, prefix=f".{path.name}.", suffix=".pending"
  )
  temporary = Path(name)
  try:
    with os.fdopen(descriptor, "wb") as stream:
      torch.save(packet, stream)
      stream.flush()
      os.fsync(stream.fileno())
    # 同目录硬链接一次性发布完整文件；目标已存在时原子拒绝，绝不覆盖。
    os.link(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)
  directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
  try:
    os.fsync(directory)
  finally:
    os.close(directory)
  return packet["teacher_id"]


def load_bundle(path, device="cpu"):
  packet = torch.load(path, map_location=device, weights_only=True)
  if packet["contract"] != CONTRACT:
    raise ValueError("共享actor的输入约定不一致")
  him, actor, cnn, gru = make_modules()
  teacher = Teacher(him, actor).to(device)
  cnn, gru = cnn.to(device), gru.to(device)
  teacher.initialization = copy.deepcopy(packet["initialization"])
  teacher.load_state_dict(packet["teacher"], strict=True)
  cnn.load_state_dict(packet["cnn"], strict=True)
  gru.load_state_dict(packet["gru"], strict=True)
  if teacher_identity(teacher) != packet["teacher_id"]:
    raise ValueError("共享actor/HIM/地形编码器权重身份不一致")
  return teacher, cnn, gru, packet


def require_teacher_evaluation(path, teacher_id, stage="student"):
  """从原始采集包重新评分；四个手填布尔不能授予资格。"""
  if path is None:
    raise ValueError("需要独立能力评估的原始证据索引")
  from eval_collect import source_files
  from eval_protocol import digest
  from qualification import require_qualification

  return require_qualification(
    path, teacher_id, stage, source_sha256=digest(source_files())
  )
