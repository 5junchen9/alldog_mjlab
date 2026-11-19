"""冻结教师、共享actor/HIM，仅训练学生CNN+GRU。"""

import torch
from model import DepthContract, Teacher, preprocess_raw_depth
from torch import nn
from torch.nn import functional as F

CONTROL_SECONDS = 0.02


class SharedStudent(nn.Module):
  def __init__(self, teacher: Teacher, depth_encoder, gru):
    super().__init__()
    teacher.freeze()
    self.teacher = teacher
    self.depth_encoder = depth_encoder.requires_grad_(True)
    self.gru = gru.requires_grad_(True)

  @property
  def actor(self):
    return self.teacher.actor

  @property
  def him(self):
    return self.teacher.him

  def train(self, mode=True):
    super().train(mode)
    self.teacher.freeze()
    return self

  def student_parameters(self):
    """新学生optimizer只能接收这些参数。"""
    return list(self.depth_encoder.parameters()) + list(self.gru.parameters())

  def act_with_visual(self, history, visual):
    # actor参数冻结，但这里保留visual的计算图，让动作损失训练视觉分支。
    features = torch.cat((history[:, -1].detach(), self.him(history), visual), dim=-1)
    return self.actor(features)

  def forward(self, history, depth64, metadata, state=None):
    """metadata使用原GRU的tick/episode/capture_episode/frame/capture_tick/mask。"""
    # 以当前控制步为时间原点，保留真实帧龄，避免长时钟转浮点秒后相减丢精度。
    capture_s = (metadata["capture_tick"] - metadata["tick"]).to(
      torch.float64
    ) * CONTROL_SECONDS
    control_s = torch.zeros_like(capture_s)
    same_episode = (
      (metadata["capture_episode"] == metadata["episode"])
      & (metadata["frame"] >= 0)
      & metadata["mask"]
    )
    image = preprocess_raw_depth(
      depth64, capture_s, control_s, same_episode, DepthContract(64, 48)
    )
    encoded = self.depth_encoder(image)
    visual, state = self.gru(
      encoded,
      tick=metadata["tick"],
      episode=metadata["episode"],
      capture_episode=metadata["capture_episode"],
      frame=metadata["frame"],
      capture_tick=metadata["capture_tick"],
      mask=metadata["mask"],
      state=state,
    )
    return self.act_with_visual(history, visual), visual, state

  def forward_sequence(self, histories, depths, metadata, state=None):
    """按真实连续控制步回放；GRU自行检查时钟，并在episode变化时清隐状态。"""
    if (
      not len(metadata)
      or len(histories) != len(metadata)
      or len(depths) != len(metadata)
    ):
      raise ValueError("aligned nonempty control sequence required")
    actions, features = [], []
    for history, depth, row in zip(histories, depths, metadata, strict=True):
      action, visual, state = self(history, depth, row, state)
      actions.append(action)
      features.append(visual)
    return torch.stack(actions), torch.stack(features), state


def distillation_loss(history, geometry, visual, student, teacher, feature_weight):
  """两个Huber项；feature_weight显式给定，首版1.0是未调优实验值。"""
  if teacher is not student.teacher:
    raise ValueError("distillation target must be the shared frozen teacher")
  with torch.no_grad():
    teacher_features = teacher.features(history, geometry)
    target_visual = teacher_features[:, -32:]
    target_action = teacher.actor(teacher_features)
  prediction = student.act_with_visual(history, visual)
  action_loss = F.huber_loss(prediction, target_action, delta=1.0)
  feature_loss = F.huber_loss(visual, target_visual, delta=1.0)
  loss = action_loss + feature_weight * feature_loss
  return loss, dict(
    action_huber=action_loss.detach(),
    feature_huber=feature_loss.detach(),
    feature_weight=float(feature_weight),
  )
