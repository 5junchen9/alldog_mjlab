"""Formal methods runner with explicit profile consistency and budget checks."""

from typing import TYPE_CHECKING, cast

from .runner import WolfHimOnPolicyRunner

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def validate_methods_profile(env_cfg, train_cfg: dict) -> None:
  actor = train_cfg["actor"]
  algorithm = train_cfg["algorithm"]
  if algorithm.get("dual_critic_enabled", False):
    placement = env_cfg.rewards.get("wheel_support_placement")
    if placement is None or placement.weight == 0:
      raise ValueError("Dual critic requires an active wheel-support reward channel")
  if actor.get("distillation_enabled", False):
    for name in (actor["teacher_local_group"], actor["teacher_global_group"]):
      if name not in env_cfg.observations:
        raise ValueError(f"Terrain distillation requires observation group {name!r}")


class WolfMethodsOnPolicyRunner(WolfHimOnPolicyRunner):
  """Do not mistake tiny diagnostics for an authorized formal training round."""

  def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
    raw = cast("ManagerBasedRlEnv", getattr(env, "unwrapped", env))
    validate_methods_profile(raw.cfg, train_cfg)
    super().__init__(env, train_cfg, log_dir, device, **kwargs)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len=False):
    if num_learning_iterations < 2000:
      raise ValueError("Formal Wolf methods training requires at least 2000 iterations")
    return super().learn(num_learning_iterations, init_at_random_ep_len)
