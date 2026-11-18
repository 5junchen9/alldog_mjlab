"""同一教师内混合学习前进、后退、纯转向和纯侧移。"""

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class OmniVelocityCommand(UniformVelocityCommand):
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    probabilities = torch.tensor(
      [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.15, 0.05], device=self.device
    )
    category = torch.multinomial(probabilities, len(env_ids), replacement=True)
    commands = self.vel_command_b[env_ids].clone()
    # 前进、后退、左转、右转、左移、右移；第6类保留组合指令。
    for index, axis, sign in (
      (0, 0, 1),
      (1, 0, -1),
      (2, 2, 1),
      (3, 2, -1),
      (4, 1, 1),
      (5, 1, -1),
    ):
      selected = category == index
      speed = commands[selected, axis].abs().clamp(min=0.1)
      commands[selected] = 0.0
      commands[selected, axis] = sign * speed
    commands[category == 7] = 0.0
    self.vel_command_b[env_ids] = commands
    self.is_standing_env[env_ids] = category == 7
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False

  def _update_metrics(self) -> None:
    super()._update_metrics()
    # 按实际送给策略的指令分类，记录真实采样覆盖。
    vx, vy, yaw = self.command.unbind(dim=1)
    x_zero, y_zero, yaw_zero = vx.abs() < 1e-6, vy.abs() < 1e-6, yaw.abs() < 1e-6
    groups = {
      "forward": (vx > 0) & y_zero & yaw_zero,
      "reverse": (vx < 0) & y_zero & yaw_zero,
      "turn_left": x_zero & y_zero & (yaw > 0),
      "turn_right": x_zero & y_zero & (yaw < 0),
      "side_left": x_zero & (vy > 0) & yaw_zero,
      "side_right": x_zero & (vy < 0) & yaw_zero,
      "standing": x_zero & y_zero & yaw_zero,
    }
    pure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    for name, selected in groups.items():
      pure |= selected
      self._env.extras["log"][f"CommandCoverage/{name}"] = selected.float().mean()
    self._env.extras["log"]["CommandCoverage/mixed"] = (~pure).float().mean()


@dataclass(kw_only=True)
class OmniVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> OmniVelocityCommand:
    return OmniVelocityCommand(self, env)
