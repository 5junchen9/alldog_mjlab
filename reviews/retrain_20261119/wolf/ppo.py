"""Wolf专用观察与动作宽度；复用严格末帧timeout处理。"""

import torch
from black.ppo import TerminalPPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage

from wolf.model import ACTION_DIM, ACTOR_DIM, HISTORY_SHAPE


class CourseNormalizer(torch.nn.Module):
  """只更新原430维观测的统计量，末尾有界课程状态直接传给critic。"""

  def __init__(self):
    super().__init__()
    self.observation = EmpiricalNormalization(430)

  def forward(self, value):
    return torch.cat((self.observation(value[..., :430]), value[..., 430:]), -1)

  def update(self, value):
    self.observation.update(value[..., :430])


class CourseCritic(MLPModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.obs_dim not in (430, 433) or not self.obs_normalization:
      raise ValueError("Wolf critic需要原430维观测及可选course3")
    self.obs_normalizer = CourseNormalizer()


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
    critic = CourseCritic(
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
