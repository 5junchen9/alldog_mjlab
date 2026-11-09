"""Bounded, frozen cross-machine release check. Never starts training."""

import argparse
import hashlib
import json
import tempfile
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.wolf_him.methods_ppo import MethodsPPO
from mjlab.rl.wolf_him.spec import canonical_history
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.wolf.methods_registry import make_methods_profile
from mjlab.tasks.velocity.config.wolf.methods_runner import WolfMethodsOnPolicyRunner


def observation_tensor(obs, key) -> torch.Tensor:
  value = obs[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Expected tensor for {key!r}")
  return value


def state_hash(module):
  digest = hashlib.sha256()
  for name, value in sorted(module.state_dict().items()):
    digest.update(name.encode())
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()


def forbidden(*args, **kwargs) -> NoReturn:
  raise RuntimeError("Training and optimizer updates are forbidden in this check")


def verify_manifest(root):
  manifest_path = root / "snapshot_manifest.json"
  if not manifest_path.exists():
    return None
  manifest = json.loads(manifest_path.read_text())
  for item in manifest["files"]:
    path = root / item["path"]
    if item.get("symlink"):
      assert path.is_symlink() and str(path.readlink()) == item["symlink"]
    else:
      assert path.is_file(), str(path)
      assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], str(path)
  return {
    "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "files_verified": len(manifest["files"]),
  }


def check_profile(stage, mode, device):
  torch.manual_seed(20261109)
  options = {"soft_terrain": stage == "soft"}
  if mode in ("no_him", "aux_off"):
    options["use_him_estimator"] = False
  if mode == "aux_off":
    options.update(
      distillation=False,
      prediction=False,
      dual_critic=False,
      foot_placement=False,
      clf=False,
    )
  cfg, agent = make_methods_profile(**options)
  cfg.scene.num_envs = 2
  cfg.seed = 20261109
  agent.num_steps_per_env = 2
  agent.seed = 20261109
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=device))
  guards = ExitStack()
  try:
    runner = WolfMethodsOnPolicyRunner(env, asdict(agent), device=device)
    guards.enter_context(patch.object(runner, "learn", forbidden))
    alg = runner.alg
    assert isinstance(alg, MethodsPPO)
    guards.enter_context(patch.object(alg, "update", forbidden))
    guards.enter_context(patch.object(alg.actor.estimator, "update", forbidden))
    alg.eval_mode()
    obs = env.get_observations()
    before_actor, before_critic = state_hash(alg.actor), state_hash(alg.critic)
    env.episode_length_buf[0] = env.max_episode_length - 1
    terminal_seen = False
    with torch.inference_mode():
      for _ in range(2):
        action = alg.act(obs)
        obs, reward, done, extras = env.step(action)
        assert torch.isfinite(action).all() and torch.isfinite(reward).all()
        assert all(
          torch.isfinite(observation_tensor(obs, key)).all() for key in obs.keys()
        )
        alg.process_env_step(obs, reward, done, extras)
        terminal_seen |= bool(done[0])
      alg.compute_returns(obs)
      assert terminal_seen and torch.isfinite(alg.storage.advantages).all()
      history = canonical_history(observation_tensor(obs, "actor"))
      export = torch.jit.script(alg.actor.as_jit())
      expected = alg.actor(obs.select("actor"))
      torch.testing.assert_close(export(history), expected)
      if mode in ("no_him", "aux_off"):
        assert not any(key.startswith("encoder.") for key in export.state_dict())
      with tempfile.TemporaryDirectory(prefix="wolf_UNTRAINED_release_") as temporary:
        path = Path(temporary) / "UNTRAINED_FIXTURE.pt"
        runner.save(str(path))
        runner.load(str(path))
        torch.testing.assert_close(alg.actor(obs.select("actor")), expected)
      assert state_hash(alg.actor) == before_actor
      assert state_hash(alg.critic) == before_critic
    return {
      "stage": stage,
      "mode": mode,
      "num_envs": 2,
      "control_steps": 2,
      "forced_timeout_seen": terminal_seen,
      "actor_and_critic_unchanged": True,
      "jit_roundtrip": True,
      "checkpoint_roundtrip": "untrained optimizer fixture only",
      "actor_input": [6, 53],
      "export_input": 318,
      "action_dim": 16,
      "value_channels": alg.storage.values.shape[-1],
      "method_signature": alg._methods_signature(),
    }
  finally:
    guards.close()
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError("Refusing to overwrite a previous validation report")
  root = Path(__file__).resolve().parents[1]
  manifest = verify_manifest(root)
  for name in ("wolf-flat", "wolf-rough", "wolf-flat-him", "wolf-rough-him"):
    cfg, rl = load_env_cfg(name), load_rl_cfg(name)
    assert cfg.scene.num_envs == 4096 and rl.num_steps_per_env == 100
    assert "wheel_support_placement" not in cfg.rewards
    assert "terrain_teacher_local" not in cfg.observations
  profiles = [
    ("soft", "full"),
    ("hard", "full"),
    ("soft", "no_him"),
    ("soft", "aux_off"),
    ("hard", "aux_off"),
  ]
  with (
    patch.object(torch.Tensor, "backward", forbidden),
    patch.object(torch.optim.Adam, "step", forbidden),
  ):
    reports = [check_profile(stage, mode, args.device) for stage, mode in profiles]
  result = {
    "training_started": False,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "source_manifest": manifest,
    "versions": {
      key: version(key)
      for key in ("torch", "rsl-rl-lib", "mujoco", "mujoco-warp", "tensordict")
    },
    "gpu": torch.cuda.get_device_name() if "cuda" in args.device else None,
    "original_tasks_preserved": True,
    "profiles": reports,
    "limitations": [
      "No learned teacher or locomotion quality evaluation",
      "No actual optimizer updates or training-capacity benchmark",
      "Checkpoint checks use untrained fixtures, not learned state",
    ],
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
