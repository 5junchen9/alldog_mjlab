"""Independent, plan-first straight/turn HIM launchers for the frozen v2 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

import mjlab
import mjlab.tasks  # noqa: F401
import torch
from mjlab.asset_zoo.robots.wolf.wolf_constants import WOLF_POLICY_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.wolf.env_cfg import (
  WOLF_ACTION_DIM,
  WOLF_ACTOR_OBS_DIM,
  WOLF_COMMAND_NAME,
  WOLF_CRITIC_DIM,
)
from mjlab.tasks.velocity.config.wolf.mdp.commands import (
  WolfVelocityCommand,
  WolfVelocityCommandCfg,
)
from mjlab.tasks.velocity.config.wolf.mdp.curriculums import (
  CURRICULUM_TERM_NAME,
  WolfForwardSpeedCommandCurriculum,
)
from mjlab.tasks.velocity.config.wolf.mdp.him import WOLF_HIM_HISTORY_LENGTH
from mjlab.tasks.velocity.config.wolf.rl_cfg import WolfHimRunnerCfg

BASE_TASK = "wolf-flat-him"
Profile = Literal["straight", "turn"]
RELEASE_MANIFEST_SHA256 = (
  "11e2df1a3bf5784c81e99149459ccbffed4ea695ae39b05525908e89405e45c7"
)


class PlanarVelocityCommand(WolfVelocityCommand):
  """Project native samples to the declared pure domain, including RNG endpoints."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    cfg = cast(PlanarVelocityCommandCfg, self.cfg)
    self.vel_command_b[env_ids, 1] = 0.0
    self.vel_command_b[env_ids, 2 if cfg.profile == "straight" else 0] = 0.0
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
    self.is_world_env[env_ids] = False
    self.is_heading_env[env_ids] = False
    self.is_forward_env[env_ids] = False


@dataclass(kw_only=True)
class PlanarVelocityCommandCfg(WolfVelocityCommandCfg):
  profile: Profile = "straight"

  def build(self, env: ManagerBasedRlEnv) -> PlanarVelocityCommand:
    return PlanarVelocityCommand(self, env)


@dataclass(frozen=True)
class Options:
  profile: Profile
  num_envs: int = 4096
  rollout: int = 100
  epochs: int = 5
  minibatches: int = 8
  seed: int | None = None
  iterations: int = 2000
  save_interval: int = 50
  log_root: str = str(Path(__file__).resolve().parent / "logs")
  checkpoint: str | None = None
  train: bool = False


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def validate_options(options: Options) -> None:
  if options.profile not in ("straight", "turn"):
    raise ValueError("profile must be straight or turn")
  if not 1 <= options.num_envs <= 4096:
    raise ValueError("num-envs must be 1..4096; higher counts need new admission")
  if options.rollout != 100 or options.epochs != 5:
    raise ValueError("The admitted budget fixes rollout=100 and epochs=5")
  if options.minibatches not in (4, 8):
    raise ValueError("minibatches must be 4 or 8")
  if options.num_envs * options.rollout % options.minibatches:
    raise ValueError("rollout batch must divide evenly across minibatches")
  if options.iterations < 2000:
    raise ValueError("iterations must be >=2000; this is not a smoke-run launcher")
  if options.save_interval != 50:
    raise ValueError("save-interval must remain 50")
  if options.seed is not None and not 0 <= options.seed <= 2**32 - 1:
    raise ValueError("seed must be an unsigned 32-bit integer")
  if not options.log_root.strip():
    raise ValueError("log-root must not be empty")
  if options.checkpoint is not None:
    checkpoint = Path(options.checkpoint).expanduser()
    if not checkpoint.is_file() or checkpoint.suffix != ".pt":
      raise ValueError("checkpoint must identify one existing .pt file, not a regex")


def verify_release() -> dict[str, Any]:
  """Reject other source trees; do not mutate the selected frozen release."""
  root = Path(mjlab.__file__).resolve().parents[2]
  manifest_path = root / "snapshot_manifest.json"
  data = manifest_path.read_bytes()
  if hashlib.sha256(data).hexdigest() != RELEASE_MANIFEST_SHA256:
    raise ValueError("The imported mjlab is not the pinned v2 release")
  manifest = json.loads(data)
  for item in manifest["files"]:
    path = root / item["path"]
    if item.get("symlink"):
      valid = path.is_symlink() and str(path.readlink()) == item["symlink"]
    else:
      valid = path.is_file() and (
        hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
      )
    if not valid:
      raise ValueError(f"Frozen release verification failed: {path}")
  return {
    "root": str(root),
    "release": manifest["release"],
    "manifest_sha256": RELEASE_MANIFEST_SHA256,
    "files_verified": len(manifest["files"]),
  }


def build_training(options: Options) -> TrainConfig:
  validate_options(options)
  env = load_env_cfg(BASE_TASK)
  agent = load_rl_cfg(BASE_TASK)
  if not isinstance(agent, WolfHimRunnerCfg):
    raise TypeError("Expected the native Wolf HIM runner configuration")
  command = env.commands[WOLF_COMMAND_NAME]
  if not isinstance(command, WolfVelocityCommandCfg):
    raise TypeError("Expected native Wolf velocity command sampling")
  curriculum = env.curriculum.pop(CURRICULUM_TERM_NAME, None)
  if curriculum is None or curriculum.func is not WolfForwardSpeedCommandCurriculum:
    raise ValueError("Unexpected source command curriculum; review before training")
  if env.curriculum:
    raise ValueError("Flat specialist must have no hidden curriculum terms")
  command.ranges.lin_vel_x = (
    (-1.0, 1.0) if options.profile == "straight" else (0.0, 0.0)
  )
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (
    (0.0, 0.0) if options.profile == "straight" else (-1.5, 1.5)
  )
  command.ranges.heading = None
  command.heading_command = False
  command.rel_heading_envs = 0.0
  command.rel_world_envs = 0.0
  command.rel_forward_envs = 0.0
  command.rel_standing_envs = 0.1
  # Native zero-probability branches use <=; project rare exact-zero RNG draws.
  env.commands[WOLF_COMMAND_NAME] = PlanarVelocityCommandCfg(
    **{field.name: getattr(command, field.name) for field in fields(command)},
    profile=options.profile,
  )
  # Preserve native small_xy_threshold=0.2, physics, reward, observations and HIM.
  env.scene.num_envs = options.num_envs
  env.seed = (
    options.seed
    if options.seed is not None
    else (301 if options.profile == "straight" else 401)
  )
  agent.seed = env.seed
  agent.num_steps_per_env = options.rollout
  agent.algorithm.num_learning_epochs = options.epochs
  agent.algorithm.num_mini_batches = options.minibatches
  agent.max_iterations = options.iterations
  agent.save_interval = options.save_interval
  agent.logger = "tensorboard"
  agent.upload_model = False
  agent.experiment_name = f"wolf_planar_{options.profile}_him_20261109"
  agent.run_name = f"wolf_planar_{options.profile}_him"
  agent.command_curriculum_restore = "none"
  agent.resume = False
  agent.warm_start = False
  agent.checkpoint_path = (
    str(Path(options.checkpoint).expanduser().resolve())
    if options.checkpoint is not None
    else None
  )
  return TrainConfig(
    env=env,
    agent=agent,
    log_root=str(Path(options.log_root).expanduser().resolve()),
    gpu_ids=[0],
  )


def domain_contract(
  options: Options, cfg: TrainConfig, release: dict[str, Any]
) -> dict[str, Any]:
  command = cfg.env.commands[WOLF_COMMAND_NAME]
  assert isinstance(command, WolfVelocityCommandCfg)
  agent = cast(WolfHimRunnerCfg, cfg.agent)
  root = Path(release["root"])
  source_contract = json.loads((root / "docs/wolf/source_contract.json").read_text())
  straight = options.profile == "straight"
  return {
    "schema_version": 1,
    "status": "training_configuration_only_not_a_certified_teacher",
    "profile": options.profile,
    "base_task": BASE_TASK,
    "source": release,
    "command": {
      "name": WOLF_COMMAND_NAME,
      "linear_frame": "body",
      "angular_axis": "body_z",
      "units": ["m/s", "m/s", "rad/s"],
      "native_cfg": asdict(command),
      "sampling": "native_uniform_deadband_then_pure_domain_projection_and_standing",
      "projection": "vy=0; straight:yaw=0; turn:vx=0; world/heading/forward flags cleared",
      "endpoint_note": "Native <=0 RNG branch can add rare zero yaw; projection prevents off-domain commands",
      "deadband_rule": "sampled_xy_norm <= small_xy_threshold => xy=(0,0)",
      "nonzero_vx_support": [[-1.0, -0.2], [0.2, 1.0]] if straight else [],
      "nonzero_vx_inner_endpoints_excluded": straight,
      "explicit_standing_probability": 0.1,
      "expected_total_zero_command_probability": 0.28 if straight else 0.1,
      "expected_negative_command_probability": 0.36 if straight else 0.45,
      "expected_positive_command_probability": 0.36 if straight else 0.45,
      "probability_note": "Continuous-uniform expectation; finite RNG may differ",
    },
    "curriculum": {
      "removed": [CURRICULUM_TERM_NAME],
      "remaining": list(cfg.env.curriculum),
      "restore": agent.command_curriculum_restore,
      "command_domain_frozen": True,
    },
    "policy_contract": {
      "single_frame_dim": WOLF_ACTOR_OBS_DIM,
      "history_frames": WOLF_HIM_HISTORY_LENGTH,
      "history_dim": WOLF_ACTOR_OBS_DIM * WOLF_HIM_HISTORY_LENGTH,
      "history_tensor_shape": [WOLF_HIM_HISTORY_LENGTH, WOLF_ACTOR_OBS_DIM],
      "critic_dim": WOLF_CRITIC_DIM,
      "action_dim": WOLF_ACTION_DIM,
      "action_terms": list(cfg.env.actions),
      "policy_joint_order": list(WOLF_POLICY_JOINT_NAMES),
      "observation_terms": list(cfg.env.observations["actor"].terms),
      "control": source_contract["control"],
      "him": asdict(agent.him),
      "actor": asdict(agent.actor),
      "critic": asdict(agent.critic),
      "physics_rewards_randomization": "unchanged_from_pinned_base_task",
      "terrain": "plane",
      "camera": None,
    },
    "training": {
      "num_envs": cfg.env.scene.num_envs,
      "rollout": agent.num_steps_per_env,
      "epochs": agent.algorithm.num_learning_epochs,
      "minibatches": agent.algorithm.num_mini_batches,
      "transitions_per_iteration": options.num_envs * options.rollout,
      "transitions_per_minibatch": (
        options.num_envs * options.rollout // options.minibatches
      ),
      "requested_learning_iterations": agent.max_iterations,
      "requested_transitions": options.num_envs * options.rollout * options.iterations,
      "seed": agent.seed,
      "save_interval": agent.save_interval,
      "experiment_name": agent.experiment_name,
      "run_name": agent.run_name,
      "log_root": str(Path(cfg.log_root).expanduser().resolve()),
      "logger": agent.logger,
      "upload_model": agent.upload_model,
      "gpu_ids": cfg.gpu_ids,
      "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
      "admitted_allocator": "expandable_segments:True",
      "algorithm": asdict(agent.algorithm),
      "exact_checkpoint": agent.checkpoint_path,
      "checkpoint_source_sha256": (
        file_sha256(Path(agent.checkpoint_path)) if agent.checkpoint_path else None
      ),
      "resume_mechanism": "native_checkpoint_path_full_HIM_state_not_warm_start",
      "checkpoint_domain_provenance": "not_checked_by_this_launcher",
      "cross_profile_resume": "not_guaranteed_semantically_equivalent",
      "train_requested": options.train,
    },
    "limitations": [
      "Specialists only: no combined vx+yaw or lateral command coverage",
      "No terrain/camera teacher claim and no deployment certification",
      "Configuration/sample checks are not a convergence or GPU admission test",
      "Pair the printed domain with exact checkpoint path/hash when making a sidecar",
    ],
  }


def parse_options(argv: list[str] | None = None) -> Options:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--profile", required=True, choices=("straight", "turn"))
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--rollout", type=int, default=100, choices=(100,))
  parser.add_argument("--epochs", type=int, default=5, choices=(5,))
  parser.add_argument("--minibatches", type=int, default=8, choices=(4, 8))
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--iterations", type=int, default=2000)
  parser.add_argument("--save-interval", type=int, default=50, choices=(50,))
  parser.add_argument(
    "--log-root", default=Options.__dataclass_fields__["log_root"].default
  )
  parser.add_argument(
    "--checkpoint", help="Exact existing HIM .pt path; no regex lookup"
  )
  parser.add_argument("--train", action="store_true", help="Start training explicitly")
  options = Options(**vars(parser.parse_args(argv)))
  try:
    validate_options(options)
  except ValueError as error:
    parser.error(str(error))
  return options


def main(argv: list[str] | None = None) -> int:
  options = parse_options(argv)
  release = verify_release()
  cfg = build_training(options)
  print(json.dumps(domain_contract(options, cfg, release), indent=2), flush=True)
  if options.train:
    # Keep the existing registry task ID so its native HIM runner is selected.
    launch_training(task_id=BASE_TASK, args=cfg)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
