"""Wolf专用观察与动作宽度；复用严格末帧timeout处理。"""

import torch
from black.ppo import TerminalPPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from wolf.model import ACTION_DIM, ACTOR_DIM, HISTORY_SHAPE


class PPOActor(MLPModel):
  def __init__(self, obs, teacher):
    super().__init__(
      obs,
      {"actor": ["history", "geometry"]},
      "actor",
      ACTION_DIM,
      (512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
        "std_range": (0.05, 2.0),
      },
    )
    self.policy = teacher
    self.mlp = teacher.actor

  def _get_latent_dim(self):
    return ACTOR_DIM

  def get_latent(self, obs, masks=None, hidden_state=None):
    return self.policy.features(
      obs["history"].reshape(-1, *HISTORY_SHAPE), obs["geometry"]
    )


def build_ppo(obs, teacher, *, num_steps=24, device="cpu"):
  if any(p.requires_grad for p in teacher.him.parameters()):
    raise ValueError("PPO阶段HIM必须冻结")
  actor = PPOActor(obs, teacher).to(device)
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(teacher.initialization["critic_seed"])
    critic = MLPModel(
      obs,
      {"critic": ["critic", "course_state"] if "course_state" in obs else ["critic"]},
      "critic",
      1,
      (512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ).to(device)
  storage = RolloutStorage("rl", len(obs), num_steps, obs, [ACTION_DIM], device)
  algorithm = TerminalPPO(
    actor,
    critic,
    storage,
    device=device,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1e-3,
    schedule="adaptive",
    desired_kl=0.01,
  )
  for group in algorithm.optimizer.param_groups:
    group["params"] = [p for p in group["params"] if p.requires_grad]
  return algorithm
