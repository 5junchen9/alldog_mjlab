"""Wolf104维共享actor，HIM与PPO交替更新，学生只训练视觉。"""

import torch
from torch import nn

from mjlab.rl.wolf_him.estimator import HIMEstimator
from mjlab.rl.wolf_him.spec import HIMSpec, canonical_history, estimator_target_input

HISTORY_SHAPE = (6, 53)
ACTION_DIM = 16
ACTOR_DIM = 104
VELOCITY_SCALE = 2.0
SPEC = HIMSpec(53, 6, 3, 3, 16, 16)


class HIM(HIMEstimator):
  def __init__(self):
    super().__init__(SPEC)
    self.freeze()

  def freeze(self):
    self.requires_grad_(False)
    super().train(False)
    for parameter in self.parameters():
      parameter.grad = None
    return self

  def train(self, mode=True):
    return super().train(False)

  def enable_learning(self):
    self.requires_grad_(True)
    super().train(True)

  def forward(self, history):
    with torch.no_grad():
      velocity, latent = self.encode(canonical_history(history))
      return torch.cat((velocity, latent), -1)

  def estimator_losses(self, history, next_frame, next_velocity):
    return self.losses(
      canonical_history(history),
      estimator_target_input(next_frame, next_velocity, SPEC),
      next_velocity,
    )


class Teacher(nn.Module):
  history_shape = HISTORY_SHAPE

  def __init__(self):
    super().__init__()
    self.him = HIM()
    self.terrain_encoder = nn.Sequential(
      nn.Linear(768, 128), nn.ELU(), nn.Linear(128, 32)
    )
    self.actor = nn.Sequential(
      nn.Linear(ACTOR_DIM, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, ACTION_DIM),
    )
    self.initialization = {
      "mode": "scratch",
      "seed": torch.initial_seed(),
      "critic_seed": torch.initial_seed() + 1,
    }

  def features(self, history, geometry):
    if history.shape[1:] != HISTORY_SHAPE:
      raise ValueError("Wolf本体历史必须为[B,6,53]")
    return torch.cat(
      (history[:, -1].detach(), self.him(history), self.terrain_encoder(geometry)), -1
    )

  def forward(self, history, geometry):
    return self.actor(self.features(history, geometry))

  def freeze(self):
    self.eval().requires_grad_(False)
    for parameter in self.parameters():
      parameter.grad = None
    return self


def make_student_modules():
  from support.black.causal_visual_gru import CausalVisualGRU
  from support.black.depth_him import make_depth_encoder

  return make_depth_encoder(), CausalVisualGRU(enabled=True)


def initial_models(seed):
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(seed)
    teacher = Teacher()
    cnn, gru = make_student_modules()
  return teacher, cnn, gru
