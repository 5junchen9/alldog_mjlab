"""One standard PPO actor/critic with per-profile diagnostics, no distillation."""

import torch
from rsl_rl.algorithms import PPO


class TaskLoggedPPO(PPO):
  """Preserve upstream PPO math; profile IDs are used for logs only."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.profile_id: torch.Tensor | None = None
    self.profile_metrics = {}

  def compute_returns(self, obs):
    super().compute_returns(obs)
    if self.profile_id is None:
      return
    for index, name in enumerate(("low", "omni", "high")):
      mask = self.profile_id == index
      if mask.any():
        values = self.storage.returns[:, mask]
        advantages = self.storage.advantages[:, mask]
        self.profile_metrics[f"{name}_return_mean"] = values.mean().item()
        self.profile_metrics[f"{name}_advantage_mean"] = advantages.mean().item()
        self.profile_metrics[f"{name}_advantage_std"] = advantages.std(
          unbiased=False
        ).item()

  def update(self):
    result = super().update()
    result.update(self.profile_metrics)
    for module in (self.actor, self.critic):
      if any(not torch.isfinite(p).all() for p in module.parameters()):
        raise FloatingPointError("Nonfinite actor/critic after PPO update")
    return result
