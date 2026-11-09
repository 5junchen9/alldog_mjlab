"""Export a loaded Methods actor for a separate, frozen-teacher DAgger stage.

Export parity proves a software bridge, not expert quality. Every generated
manifest remains UNQUALIFIED until independent closed-loop validation exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.wolf_him.perception_models import PerceptionHIMPolicy
from mjlab.rl.wolf_him.spec import canonical_history
from mjlab.tasks.registry import load_runner_cls
from mjlab.tasks.velocity.config.wolf.methods_cfg import wolf_methods_runner_cfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg

from .contracts import FrozenTeacher, TargetSpaceAdapter, adapter_digest, sha256_file
from .run import current_contract, make_config, runtime_digest

METHOD_TASKS = ("wolf-methods-soft", "wolf-methods-hard")


def parameter_digest(model: torch.nn.Module) -> str:
  digest = hashlib.sha256()
  for name, tensor in sorted(model.state_dict().items()):
    digest.update(name.encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()


@torch.no_grad()
def export_methods_actor(
  policy: PerceptionHIMPolicy,
  observations: TensorDict,
  env_cfg,
  output: Path,
  *,
  task: str,
  source_checkpoint: Path,
) -> dict:
  """Export the exact loaded actor, test same-state parity, never train/qualify.

  Source checkpoint actor tensors must match the provided actor byte-for-byte.
  This accepts only the declared methods stage and current actual controls.
  Output must be a fresh directory; source checkpoint is never overwritten.
  """
  if task not in METHOD_TASKS or not isinstance(policy, PerceptionHIMPolicy):
    raise ValueError("this bridge requires a Methods actor and explicit methods stage")
  if policy.history_group != "actor":
    raise ValueError("unsupported Methods history group")
  history = observations["actor"]
  if not isinstance(history, torch.Tensor):
    raise ValueError("actor must be a tensor")
  if history.ndim != 3 or tuple(history.shape[1:]) != (6, 53):
    raise ValueError("Methods bridge requires actor [B, 6, 53]")
  if not torch.isfinite(history).all():
    raise ValueError("nonfinite actor history")
  loaded = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
  source_state = loaded.get("actor_state_dict", {})
  actual_state = policy.state_dict()
  if loaded.get("methods_signature") is None:
    raise ValueError("source is not a Methods checkpoint")
  if set(source_state) != set(actual_state) or any(
    not torch.equal(source_state[name].cpu(), actual_state[name].cpu())
    for name in actual_state
  ):
    raise ValueError("loaded actor does not match source checkpoint exactly")
  target_cfg = make_config(task, env_cfg.scene.num_envs, env_cfg.seed)
  # TerrainGenerator fills each subterrain size in place on construction.
  # Compare the effective geometry rather than pre/post-build placeholder sizes.
  source_terrain = asdict(env_cfg.scene.terrain.terrain_generator)
  target_entity = cast(TerrainEntityCfg, target_cfg.scene.terrain)
  target_terrain = asdict(cast(TerrainGeneratorCfg, target_entity.terrain_generator))
  for descriptor in (source_terrain, target_terrain):
    for terrain in descriptor["sub_terrains"].values():
      terrain["size"] = descriptor["size"]
  if source_terrain != target_terrain or asdict(env_cfg.commands["twist"]) != asdict(
    target_cfg.commands["twist"]
  ):
    raise ValueError("methods terrain/command domain does not match target stage")
  source_obs, target_obs = (
    env_cfg.observations["actor"],
    target_cfg.observations["actor"],
  )
  if (
    source_obs.history_length != target_obs.history_length
    or source_obs.flatten_history_dim != target_obs.flatten_history_dim
    or tuple(source_obs.terms) != tuple(target_obs.terms)
    or any(
      source_obs.terms[key].func != target_obs.terms[key].func
      or source_obs.terms[key].scale != target_obs.terms[key].scale
      for key in source_obs.terms
    )
  ):
    raise ValueError("Methods actor observation contract mismatch")
  control = current_contract(env_cfg)
  if current_contract(target_cfg) != control:
    raise ValueError("source/target controls differ; use an audited contract bridge")
  # Avoid changing the caller's training flags, requires_grad or normalizers.
  before = parameter_digest(policy)
  frozen = copy.deepcopy(policy).eval().requires_grad_(False)
  scripted = torch.jit.script(frozen.as_jit().eval()).to(history.device)
  canonical = canonical_history(history.detach().clone())
  expected = frozen(observations)
  actual = scripted(canonical)
  torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
  if actual.shape != (history.shape[0], 16) or not torch.isfinite(actual).all():
    raise ValueError("Methods export must produce finite [B, 16] actions")
  # Prove scans/critic/velocity targets are not part of this action expert.
  poisoned = observations.clone()
  for key in poisoned.keys():
    if key != "actor":
      poisoned[key] = torch.full_like(poisoned[key], float("nan"))
  torch.testing.assert_close(frozen(poisoned), expected, atol=0, rtol=0)
  adapter = FrozenTeacher(
    scripted,
    TargetSpaceAdapter(control, control),
    torch.eye(53),
    torch.zeros(53),
    history_length=6,
    newest_first=True,
  ).to(history.device)
  torch.testing.assert_close(adapter(history), expected, atol=1e-6, rtol=1e-5)
  if before != parameter_digest(policy):
    raise RuntimeError("export unexpectedly changed source actor")
  output.mkdir(parents=True, exist_ok=False)
  actor_path = output / "methods_actor.pt"
  scripted.save(str(actor_path))
  manifest = {
    "schema_version": 1,
    "qualification": "unqualified",
    "format": "torchscript_tensor",
    "teacher_sha256": sha256_file(actor_path),
    "source_checkpoint": str(source_checkpoint.resolve()),
    "source_checkpoint_sha256": sha256_file(source_checkpoint),
    "source_actor_state_sha256": before,
    "source_training_task": "not_verified_by_checkpoint",
    "source_methods_signature": loaded["methods_signature"],
    "source_control": asdict(control),
    "target_task": task,
    "target_contract_sha256": control.digest(),
    "target_runtime_sha256": runtime_digest(),
    "input_group": "actor",
    "history_length": 6,
    "history_order": "newest_first",
    "observation_matrix": torch.eye(53).tolist(),
    "observation_bias": [0.0] * 53,
    "observation_contract": "Current Wolf actor frame 53; no privileged groups",
    "normalization_contract": "Methods export embeds enabled encoders; actor normalization off",
    "history_contract": "[B,6,53] oldest-first -> canonical [B,318] newest-first",
    "curriculum_policy": "frozen_initial_config",
    "evidence": {},
  }
  report = {
    "qualification": "unqualified",
    "training_started": False,
    "optimizer_steps": 0,
    "task": task,
    "batch_size": history.shape[0],
    "export_max_absolute_error": float((expected - actual).abs().max()),
    "privileged_group_poisoning_unchanged": True,
    "source_actor_unchanged": True,
    "adapter_sha256": adapter_digest(manifest),
    "note": "Export parity only. No locomotion or teacher-quality qualification.",
  }
  (output / "teacher_manifest_UNQUALIFIED.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
  )
  (output / "export_parity.json").write_text(json.dumps(report, indent=2) + "\n")
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True, choices=METHOD_TASKS)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--him", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
    "--distillation", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument(
    "--prediction", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument(
    "--dual-critic", action=argparse.BooleanOptionalAction, default=True
  )
  args = parser.parse_args()
  cfg = make_config(args.task, num_envs=1, seed=101)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device))
  try:
    rc = asdict(
      wolf_methods_runner_cfg(
        use_him_estimator=args.him,
        soft_terrain=args.task == "wolf-methods-soft",
        distillation=args.distillation,
        prediction=args.prediction,
        dual_critic=args.dual_critic,
      )
    )
    rc.update(
      upload_model=False, logger="tensorboard", command_curriculum_restore="none"
    )
    runner_cls = load_runner_cls(args.task)
    if runner_cls is None:
      raise ValueError("Methods task has no registered runner")
    runner = runner_cls(env, rc, log_dir=None, device=args.device)
    # Full strict algorithm load also checks Methods and dual-critic signatures.
    runner.load(str(args.checkpoint), strict=True, map_location=args.device)
    obs, _ = env.reset()
    actor = runner.alg.actor
    if not isinstance(actor, PerceptionHIMPolicy):
      raise ValueError("loaded runner is not a Methods actor")
    report = export_methods_actor(
      actor,
      obs,
      cfg,
      args.output,
      task=args.task,
      source_checkpoint=args.checkpoint,
    )
    print(json.dumps(report, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
