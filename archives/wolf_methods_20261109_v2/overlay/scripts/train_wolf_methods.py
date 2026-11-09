"""Matched Wolf method ablations. Default is description only, never training.

Example: uv run --no-sync python scripts/train_wolf_methods.py --stage soft
Only --train enters the formal >=2000-iteration learning path.
"""

import argparse
import json

import mjlab.tasks  # noqa: F401
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.velocity.config.wolf.methods_registry import make_methods_profile


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", choices=("soft", "hard"), default="soft")
  parser.add_argument("--no-terrain-distillation", action="store_true")
  parser.add_argument("--no-prediction", action="store_true")
  parser.add_argument("--no-him", action="store_true")
  parser.add_argument("--no-dual-critic", action="store_true")
  parser.add_argument("--no-foot-placement", action="store_true")
  parser.add_argument("--no-clf", action="store_true")
  parser.add_argument("--num-envs", type=int)
  parser.add_argument("--rollout-steps", type=int)
  parser.add_argument("--seed", type=int, default=101)
  parser.add_argument("--max-iterations", type=int, default=2000)
  parser.add_argument("--checkpoint")
  parser.add_argument("--log-root", default="logs/wolf_methods")
  parser.add_argument("--train", action="store_true")
  args = parser.parse_args()
  profile = {
    "soft_terrain": args.stage == "soft",
    "distillation": not args.no_terrain_distillation,
    "prediction": not args.no_prediction,
    "use_him_estimator": not args.no_him,
    "foot_placement": not args.no_foot_placement,
    "dual_critic": not (args.no_dual_critic or args.no_foot_placement),
    "clf": not args.no_clf,
  }
  env, agent = make_methods_profile(**profile)
  if args.num_envs is not None:
    if args.num_envs < 1:
      raise ValueError("num-envs must be positive")
    env.scene.num_envs = args.num_envs
  if args.rollout_steps is not None:
    if args.rollout_steps < 1:
      raise ValueError("rollout-steps must be positive")
    agent.num_steps_per_env = args.rollout_steps
  env.seed = agent.seed = args.seed
  agent.max_iterations = args.max_iterations
  agent.checkpoint_path = args.checkpoint
  task = f"wolf-methods-{args.stage}"
  print(
    json.dumps(
      {
        "task": task,
        "training_requested": args.train,
        "profile": profile,
        "num_envs": env.scene.num_envs,
        "rollout_steps": agent.num_steps_per_env,
        "max_iterations": agent.max_iterations,
        "seed": args.seed,
        "transitions_per_round": env.scene.num_envs
        * agent.num_steps_per_env
        * agent.max_iterations,
        "external_upload": False,
        "note": "Specialist forward task. Independent DAgger is a later phase, not part of this PPO update.",
      },
      indent=2,
    )
  )
  if not args.train:
    return
  if args.max_iterations < 2000:
    raise ValueError("Formal training requires at least 2000 iterations")
  launch_training(task, TrainConfig(env=env, agent=agent, log_root=args.log_root))


if __name__ == "__main__":
  main()
