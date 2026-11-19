"""新版本的成套权重；旧r800只提供初始化，不携带旧优化器。"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from model import Teacher, make_initial_modules, make_modules

HERE = Path(__file__).resolve().parent
SAMPLE = HERE.parent / "black_reliability_20261119/evidence/WINDOWS_R800_SAMPLE.pt"
SAMPLE_SHA = "bc274a3a8034cadca51d51aa5b3c64fb027b5a5673be4480d7f9c56e01962982"
CONTRACT = {
  "schema": "black_shared_actor_v1",
  "input_order": ["current_p45", "HIM_next_velocity3", "HIM_unit16", "terrain32"],
  "history_shape": [5, 45],
  "geometry": "same_depth64_fixed_pixels_optical_xyz_div4_valid_12x16",
  "depth": "planar_z_metres_nearest160x90_to64x48",
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
  if hashlib.sha256(SAMPLE.read_bytes()).hexdigest() != SAMPLE_SHA:
    raise ValueError("r800初始化来源不匹配")
  saved = torch.load(SAMPLE, map_location="cpu", weights_only=True)
  him, actor, cnn, gru = make_initial_modules(saved["learner"])
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
    "initial_source_sha256": SAMPLE_SHA,
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
  teacher.load_state_dict(packet["teacher"], strict=True)
  cnn.load_state_dict(packet["cnn"], strict=True)
  gru.load_state_dict(packet["gru"], strict=True)
  if teacher_identity(teacher) != packet["teacher_id"]:
    raise ValueError("共享actor/HIM/地形编码器权重身份不一致")
  return teacher, cnn, gru, packet


def require_teacher_evaluation(path, teacher_id):
  """完整评估协议接入前保持关闭，不能用四个手填布尔代替能力证据。"""
  if path is None:
    raise ValueError("学生训练前需要选定教师的独立能力评估结果")
  result = json.loads(Path(path).read_text())
  required = ("flat", "stairs", "low_speed_turn", "reverse")
  if result.get("teacher_id") != teacher_id or any(
    result.get("passed", {}).get(name) is not True for name in required
  ):
    raise ValueError("教师身份或平地/楼梯/低速转向/反转验收未通过")
  # 旧摘要没有可核验的场景覆盖、完整试验分母和均值动作协议。
  # 当前没有自动评估器，不能把身份匹配的passed摘要当成正式准入。
  raise ValueError("完整能力评估协议尚未接入，四项passed摘要不能开放学生正式训练")
