"""原生RSL PPO接口；每个minibatch重新编码保存的原始可见地形。"""

import sys
from pathlib import Path

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [
  str(BASE / "terrain446_success12335_20261112"),
  str(BASE.parents[1] / "src"),
]

from terrain446_models import TerrainModel  # noqa: E402


class PPOActor(MLPModel):
  def __init__(self, obs, teacher):
    super().__init__(
      obs,
      {"actor": ["history", "geometry"]},
      "actor",
      12,
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
    # 不重新初始化teacher.actor，保留已加载的r800动作头。
    self.policy = teacher
    self.mlp = teacher.actor

  def _get_latent_dim(self):
    return 96

  def get_latent(self, obs, masks=None, hidden_state=None):
    history = obs["history"].reshape(-1, 5, 45)
    return self.policy.features(history, obs["geometry"])

  def as_jit(self):
    raise NotImplementedError("Export the shared teacher/student input contract")

  def as_onnx(self, verbose=False):
    raise NotImplementedError("Export the shared teacher/student input contract")


def build_ppo(obs, teacher, *, num_steps=24, device="cpu", ppo_options=None):
  """obs包含history[B,225]、geometry[B,768]和critic[B,446]。

  只构建算法和空rollout，不采集环境、不执行优化器更新。
  ppo_options直接使用原生PPO参数；不接收旧优化器状态。
  """
  if obs["critic"].shape != (len(obs), 446):
    raise ValueError("critic requires the full raw446 observation")
  if not all(parameter.requires_grad for parameter in teacher.actor.parameters()):
    raise ValueError("teacher PPO requires a trainable new-version actor")
  if any(parameter.requires_grad for parameter in teacher.him.parameters()):
    raise ValueError("HIM must remain frozen during teacher PPO")

  actor = PPOActor(obs, teacher).to(device)
  critic = TerrainModel(
    obs,
    {"critic": ["critic"]},
    "critic",
    1,
    (512, 256, 128),
    activation="elu",
    obs_normalization=True,
  ).to(device)
  storage = RolloutStorage("rl", len(obs), num_steps, obs, [12], device)
  options = {} if ppo_options is None else dict(ppo_options)
  algorithm = PPO(actor, critic, storage, device=device, **options)
  # 原生PPO将冻结参数也列入Adam；新优化器为空，此处只保留实际训练参数。
  for group in algorithm.optimizer.param_groups:
    group["params"] = [p for p in group["params"] if p.requires_grad]
  return algorithm
