"""原生RSL PPO接口；每个minibatch重新编码保存的原始可见地形。"""

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from support.black.critic import TerrainModel  # noqa: E402
from tensordict import TensorDict


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
    # 不重新初始化teacher.actor，使用当前v2共享动作头。
    self.policy = teacher
    self.mlp = teacher.actor

  def _get_latent_dim(self):
    return 96

  def get_latent(self, obs, masks=None, hidden_state=None):
    history = obs["history"].reshape(-1, 5, 45)
    return self.policy.features(history, obs["geometry"])

  def as_jit(self):
    from black.export import teacher_inference_module

    return teacher_inference_module(self.policy)

  def as_onnx(self, verbose=False):
    return OnnxTeacher(self.policy, verbose)


class OnnxTeacher(torch.nn.Module):
  """RSL导出接口：独立history与geometry输入，输出确定性的均值动作。"""

  is_recurrent = False

  def __init__(self, teacher, verbose=False):
    super().__init__()
    from black.export import teacher_inference_module

    self.policy = teacher_inference_module(teacher)
    self.verbose = verbose

  def forward(self, history, geometry):
    return self.policy(history, geometry)

  def get_dummy_inputs(self):
    return (torch.zeros(1, 5, 45), torch.zeros(1, 768))

  @property
  def input_names(self):
    return ["history", "geometry"]

  @property
  def output_names(self):
    return ["actions"]


class TerminalPPO(PPO):
  """纯超时用reset前的V(s_next)，真实终止不bootstrap。"""

  def process_env_step(self, obs, rewards, dones, extras):
    adjusted = rewards.clone()
    timeouts = extras.get("time_outs")
    if timeouts is not None and bool(timeouts.any()):
      terminal = extras["terminated"].bool()
      pure_timeout = timeouts.bool() & dones.bool() & ~terminal
      if bool(pure_timeout.any()):
        terminal_obs = TensorDict(
          {"critic": extras["terminal_critic"][pure_timeout]},
          batch_size=[int(pure_timeout.sum())],
        )
        with torch.no_grad():
          value = self.critic(terminal_obs).squeeze(-1)
        adjusted[pure_timeout] += self.gamma * value
    # 原生实现使用V(s_t)近似；移除标志防止重复补偿。
    native_extras = {key: value for key, value in extras.items() if key != "time_outs"}
    super().process_env_step(obs, adjusted, dones, native_extras)


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

  # 环境构建会消耗RNG；critic初始化用单独CPU流，不随环境数漂移。
  critic_seed = teacher.initialization["critic_seed"]
  with torch.random.fork_rng(devices=[]):
    torch.set_rng_state(
      torch.Generator(device="cpu").manual_seed(critic_seed).get_state()
    )
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
  algorithm = TerminalPPO(actor, critic, storage, device=device, **options)
  # 原生PPO将冻结参数也列入Adam；新优化器为空，此处只保留实际训练参数。
  for group in algorithm.optimizer.param_groups:
    group["params"] = [p for p in group["params"] if p.requires_grad]
  return algorithm
