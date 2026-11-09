"""Small real-simulator integration check; never calls learn or an optimizer.

Example: uv run --no-sync python scripts/check_wolf_methods.py --stage soft
This verifies wiring and finite values, NOT learned locomotion performance.
"""

import argparse
import json
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.wolf_him.methods_ppo import MethodsPPO
from mjlab.rl.wolf_him.spec import canonical_history
from mjlab.tasks.velocity.config.wolf.methods_cfg import (
  wolf_methods_env_cfg,
  wolf_methods_runner_cfg,
)
from mjlab.tasks.velocity.config.wolf.runner import WolfHimOnPolicyRunner


def observation_tensor(obs, key) -> torch.Tensor:
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Expected Tensor observation for {key!r}")
  return value


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", choices=("soft", "hard"), default="soft")
  parser.add_argument("--num-envs", type=int, default=2)
  parser.add_argument("--steps", type=int, default=2)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if not 1 <= args.num_envs <= 32 or not 1 <= args.steps <= 16:
    raise ValueError("This no-training smoke is limited to 1–32 envs and 1–16 steps")
  cfg = wolf_methods_env_cfg(soft_terrain=args.stage == "soft")
  cfg.scene.num_envs = args.num_envs
  rl = wolf_methods_runner_cfg(soft_terrain=args.stage == "soft")
  rl.num_steps_per_env = args.steps
  if "cuda" in args.device:
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device))
  try:
    runner = WolfHimOnPolicyRunner(env, asdict(rl), device=args.device)
    alg = runner.alg
    assert isinstance(alg, MethodsPPO)
    obs = env.get_observations()
    shapes = {key: list(value.shape) for key, value in obs.items()}
    initial = {key: value.clone() for key, value in alg.actor.state_dict().items()}
    # Exercise the pre-reset recorder instead of silently accepting reset labels.
    env.episode_length_buf[0] = env.max_episode_length - 1
    terminal_seen = False
    actions: torch.Tensor | None = None
    with torch.inference_mode():
      for _ in range(args.steps):
        actions = alg.act(obs)
        obs, rewards, dones, extras = env.step(actions)
        assert all(
          torch.isfinite(observation_tensor(obs, key)).all() for key in obs.keys()
        )
        assert torch.isfinite(rewards).all()
        terminal_seen |= bool(dones[0])
        alg.process_env_step(obs, rewards, dones, extras)
      assert actions is not None
      alg.compute_returns(obs)
      assert torch.isfinite(alg.storage.advantages).all()
      assert terminal_seen
      history = canonical_history(observation_tensor(obs, "actor"))
      auxiliary = alg.him_actor.auxiliary.losses(
        canonical_history(observation_tensor(alg.storage.observations, "actor")[-1]),
        actions,
        alg.storage.next_estimator_input[-1],
        observation_tensor(alg.storage.observations, "terrain_teacher_local")[-1],
        observation_tensor(alg.storage.observations, "terrain_teacher_global")[-1],
      )
      assert all(torch.isfinite(value) for value in auxiliary.values())
      exported = torch.jit.script(alg.actor.as_jit())
      torch.testing.assert_close(exported(history), alg.actor(obs.select("actor")))
      for key, value in alg.actor.state_dict().items():
        torch.testing.assert_close(value, initial[key])
    print(
      json.dumps(
        {
          "status": "passed_no_training",
          "stage": args.stage,
          "num_envs": args.num_envs,
          "steps": args.steps,
          "observations": shapes,
          "forced_timeout_target_seen": terminal_seen,
          "actor_parameters": sum(p.numel() for p in alg.actor.parameters()),
          "export_parameters": sum(p.numel() for p in exported.parameters()),
          "auxiliary_losses_finite": True,
          "torch_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
            torch.device(args.device)
          )
          if "cuda" in args.device
          else None,
        }
      )
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
