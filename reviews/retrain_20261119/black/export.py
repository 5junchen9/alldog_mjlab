"""CPU TorchScript学生导出与显式部署接口；不执行硬件或ONNX占位路径。"""

import copy
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from black.checkpoint import CONTRACT, teacher_identity


class TeacherDeployment(nn.Module):
  """仅教师推理子图；geometry仍须来自同一真实相机采样。"""

  def __init__(self, teacher):
    super().__init__()
    self.encoder = copy.deepcopy(teacher.him.encoder)
    self.terrain_encoder = copy.deepcopy(teacher.terrain_encoder)
    self.actor = copy.deepcopy(teacher.actor)

  def forward(self, history: Tensor, geometry: Tensor) -> Tensor:
    proprio = torch.cat((history[:, :, :30], history[:, :, 33:]), -1).flatten(1)
    encoded = self.encoder(proprio)
    features = torch.cat(
      (
        history[:, -1],
        encoded[:, :3],
        F.normalize(encoded[:, 3:], dim=-1),
        self.terrain_encoder(geometry),
      ),
      -1,
    )
    return self.actor(features)


def teacher_inference_module(teacher):
  return TeacherDeployment(teacher).cpu().eval()


class StudentDeployment(nn.Module):
  """调用方拥有相机身份/时钟；每个新有效帧只置fresh一次，reset必须置位。

  history为旧到新的5帧p45；depth已经按训练的保近点规则降为48x64。
  输出policy均值，外部按manifest执行关节映射、延迟和PD。
  """

  def __init__(self, student):
    super().__init__()
    self.him_encoder = copy.deepcopy(student.him.encoder)
    self.depth_encoder = copy.deepcopy(student.depth_encoder)
    self.gru_cell = copy.deepcopy(student.gru.gru)
    self.projection = copy.deepcopy(student.gru.projection)
    self.actor = copy.deepcopy(student.actor)
    self.max_age = student.gru.max_age
    self.memory_frames = student.gru.memory_frames

  def forward(
    self,
    history: Tensor,
    depth_m: Tensor,
    hidden: Tensor,
    fresh: Tensor,
    reset: Tensor,
    valid_frame: Tensor,
    frame_age_controls: Tensor,
    fresh_count: Tensor,
  ) -> tuple[Tensor, Tensor, Tensor]:
    if history.dim() != 3 or history.size(1) != 5 or history.size(2) != 45:
      raise ValueError("history must be FP32 [B,5,45], oldest to newest")
    if (
      depth_m.dim() != 4
      or depth_m.size(1) != 48
      or depth_m.size(2) != 64
      or depth_m.size(3) != 1
    ):
      raise ValueError("metric depth must be FP32 [B,48,64,1]")
    if hidden.shape != (history.size(0), 64):
      raise ValueError("hidden must be FP32 [B,64]")
    if (
      history.dtype != torch.float32
      or depth_m.dtype != torch.float32
      or hidden.dtype != torch.float32
    ):
      raise ValueError("history, depth and hidden must be FP32")
    if depth_m.size(0) != history.size(0):
      raise ValueError("depth and history batch sizes differ")
    if (
      fresh.shape != (history.size(0),)
      or reset.shape != fresh.shape
      or valid_frame.shape != fresh.shape
      or frame_age_controls.shape != fresh.shape
    ):
      raise ValueError("camera metadata must contain one value per lane")
    if (
      fresh.dtype != torch.bool
      or reset.dtype != torch.bool
      or valid_frame.dtype != torch.bool
    ):
      raise ValueError("fresh/reset/valid_frame must be bool [B]")
    if frame_age_controls.dtype != torch.long:
      raise ValueError("frame_age_controls must be int64 [B]")
    if not bool(torch.isfinite(history).all()) or not bool(
      torch.isfinite(hidden).all()
    ):
      raise ValueError("nonfinite policy input")
    if bool((frame_age_controls < 0).any()):
      raise ValueError("future camera frame forbidden")
    valid = valid_frame & (frame_age_controls <= self.max_age)
    pixel_valid = torch.isfinite(depth_m) & (depth_m >= 0.2) & (depth_m <= 4.0)
    pixel_valid = pixel_valid & valid[:, None, None, None]
    value = torch.where(pixel_valid, depth_m / 4.0, 0.0)
    image = torch.cat((value, pixel_valid.to(value.dtype)), -1)
    encoded = self.depth_encoder(image.permute(0, 3, 1, 2).contiguous())
    fresh_count = torch.where(reset, 0, fresh_count)
    restart = reset | (fresh & valid & (fresh_count >= self.memory_frames))
    hidden = torch.where(restart[:, None], 0.0, hidden)
    fresh_count = torch.where(restart, 0, fresh_count) + (fresh & valid).long()
    ids = (fresh & valid).nonzero().flatten()
    if ids.numel() > 0:
      hidden = hidden.index_copy(0, ids, self.gru_cell(encoded[ids], hidden[ids]))
    visual = encoded + torch.where(valid[:, None], self.projection(hidden), 0.0)
    proprio = torch.cat((history[:, :, :30], history[:, :, 33:]), -1).flatten(1)
    him = self.him_encoder(proprio)
    features = torch.cat(
      (history[:, -1], him[:, :3], F.normalize(him[:, 3:], dim=-1), visual), -1
    )
    return self.actor(features), hidden, fresh_count


def _publish(path, content):
  descriptor, name = tempfile.mkstemp(
    dir=path.parent, prefix=f".{path.name}.", suffix=".pending"
  )
  temporary = Path(name)
  try:
    with os.fdopen(descriptor, "wb") as stream:
      stream.write(content)
      stream.flush()
      os.fsync(stream.fileno())
    os.link(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def export_student(student, path):
  """产生可独立加载的TorchScript与配套manifest；现有目标拒绝覆盖。"""
  path = Path(path)
  manifest_path = path.with_suffix(path.suffix + ".json")
  if path.exists() or manifest_path.exists():
    raise FileExistsError("export and manifest must use a new path")
  module = torch.jit.script(StudentDeployment(student).cpu().eval())
  buffer = io.BytesIO()
  torch.jit.save(module, buffer)
  content = buffer.getvalue()
  manifest = {
    "schema": "black_retrain_student_torchscript_v2",
    "teacher_id": teacher_identity(student.teacher),
    "sha256": hashlib.sha256(content).hexdigest(),
    "inputs": {
      "history": "FP32[B,5,45], oldest_to_newest, current_frame_last",
      "depth_m": "FP32[B,48,64,1], planar_z_metres, nearest_valid_raw160x90_bins",
      "hidden": "FP32[B,64], initially_zero",
      "fresh": "bool[B], true_once_per_new_eligible_camera_frame",
      "reset": "bool[B], true_before_first_action_of_each_new_episode",
      "valid_frame": "bool[B], valid_capture_in_current_episode",
      "frame_age_controls": "int64[B], control_tick_minus_capture_tick",
      "fresh_count": "int64[B], initially_zero; carry returned count",
    },
    "p45_layout": [
      "angular_velocity",
      "gravity",
      "joint_position",
      "joint_velocity",
      "command",
      "previous_action",
    ],
    "outputs": ["policy_action_mean[B,12]", "next_hidden[B,64]", "next_fresh_count[B]"],
    "action_order": CONTRACT["action_order"],
    "action_scale": CONTRACT["action_scale"],
    "action_semantics": "default_joint_position + 0.25 * delayed_policy_action",
    "control_hz": 50,
    "camera_hz": 10,
    "maximum_frame_age_controls": student.gru.max_age,
    "memory_frames": student.gru.memory_frames,
    "memory_boundary": "reset_before_51st_fresh_valid_frame",
    "depth_valid_range_m": [0.2, 4.0],
    "delay": {
      "implementation": "external_native_action_history",
      "history_slots": 4,
      "lag_controls": [0, 1, 2],
      "lag_sampling": "uniform_per_lane_on_episode_reset",
      "preserve_history_on_reset": False,
      "shift": "once_per_control_before_selecting_delayed_action",
    },
    "depth_reduction": "terrain.reduce_depth; never fixed_nearest_pixel_resize",
    "physical_deployment_validated": False,
    "onnx": "not_implemented_or_validated",
  }
  _publish(path, content)
  # manifest最后发布；缺少它的模型文件不能被load_student_export接受。
  _publish(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2).encode())
  descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)
  return manifest


def load_student_export(path, runtime_action_order):
  """部署方在执行动作前核对真实通道顺序；不能只按12维猜测兼容。"""
  path = Path(path)
  manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
  if (
    manifest["schema"] != "black_retrain_student_torchscript_v2"
    or manifest["action_order"] != CONTRACT["action_order"]
  ):
    raise ValueError("unsupported exported student contract")
  if list(runtime_action_order) != manifest["action_order"]:
    raise ValueError("runtime action order does not match exported student")
  if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
    raise ValueError("exported student checksum mismatch")
  return torch.jit.load(str(path), map_location="cpu").eval(), manifest
