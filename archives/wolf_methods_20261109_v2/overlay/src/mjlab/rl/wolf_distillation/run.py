"""Explicit Wolf distillation entry point; default invocation is preflight only.

Run with ``uv run python -m mjlab.rl.wolf_distillation.run --help``.
TorchScript is executable code: use only trusted, audited local exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.wolf.env_cfg import make_env_cfg
from mjlab.tasks.velocity.config.wolf.mdp.methods_terrain import wheel_support_coverage
from mjlab.tasks.velocity.config.wolf.methods_cfg import wolf_methods_env_cfg

from .contracts import (
  ControlContract,
  FrozenTeacher,
  TargetSpaceAdapter,
  sha256_file,
  verify_teacher_manifest,
)
from .core import (
  DaggerConfig,
  DaggerTrainer,
  DeploymentStudent,
  ValidationResult,
  student_input,
)


def current_contract(cfg) -> ControlContract:
  """Extract nominal target-space contract from the actual chosen config."""
  robot = cfg.scene.entities["robot"]
  names, kinds, scales, offsets, kp, kd = [], [], [], [], [], []
  actuators = {a.target_names_expr[0]: a for a in robot.articulation.actuators}
  for term_name, term in cfg.actions.items():
    if term.clip is not None or not term.use_default_offset:
      raise ValueError("non-default offset/clipping needs an explicit adapter")
    kind = "position" if term_name.startswith("joint_pos_") else "velocity"
    for joint in term.actuator_names:
      names.append(joint)
      kinds.append(kind)
      scales.append(term.scale[joint])
      offsets.append(robot.init_state.joint_pos[joint] if kind == "position" else 0.0)
      kp.append(actuators[joint].stiffness)
      kd.append(actuators[joint].damping)
  return ControlContract(
    tuple(names),
    tuple(kinds),
    tuple(scales),
    tuple(offsets),
    tuple(kp),
    tuple(kd),
    cfg.sim.mujoco.timestep * cfg.decimation,
  )


def runtime_digest() -> str:
  """Invalidate old qualification when Wolf environment/asset code changes."""
  root = Path(__file__).resolve().parents[2]
  h = hashlib.sha256()
  for folder in (root / "asset_zoo/robots/wolf", root / "tasks/velocity/config/wolf"):
    for path in sorted(folder.rglob("*")):
      if path.is_file() and path.suffix in (".py", ".xml", ".json"):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
  return h.hexdigest()


def make_config(task: str, num_envs: int, seed: int):
  if task in ("wolf-methods-soft", "wolf-methods-hard"):
    cfg = wolf_methods_env_cfg(soft_terrain=task == "wolf-methods-soft")
  elif task in ("wolf-flat", "wolf-rough", "wolf-flat-him", "wolf-rough-him"):
    cfg = make_env_cfg(rough="rough" in task, him_enabled=task.endswith("-him"))
  else:
    raise ValueError("unsupported Wolf distillation task or methods stage")
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  # Distillation is one fixed domain. A changing command/terrain curriculum
  # would invalidate the held-out takeover benchmark and teacher qualification.
  cfg.curriculum = {}
  return cfg


def support_evaluation(raw):
  """Read-only Methods metric. No-contact frames score zero, not perfect support."""
  manager = getattr(raw, "reward_manager", None)
  if manager is None or "wheel_support_placement" not in manager.active_terms:
    return None
  params = manager.get_term_cfg("wheel_support_placement").params
  asset = params["asset_cfg"]
  wheels = raw.scene[asset.name].data.body_link_pos_w[:, asset.body_ids]
  ray = raw.scene[params["sensor_name"]].data
  n = wheels.shape[1]
  coverage = wheel_support_coverage(
    ray.distances.reshape(raw.num_envs, n, -1),
    ray.hit_pos_w[..., 2].reshape(raw.num_envs, n, -1),
    wheels[..., 2] - params.get("wheel_radius", 0.08),
    raw.scene.terrain.env_origins[:, 2],
    soft=params["soft"],
  )
  loaded = raw.scene[params["contact_sensor"]].data.force.norm(dim=-1) > params.get(
    "force_threshold", 1.0
  )
  return (coverage * loaded).sum(-1) / loaded.sum(-1).clamp_min(1)


@torch.no_grad()
def evaluate(env, policy, seed: int, teacher_group: str | None = None):
  """One complete episode per env, same fixed validation seeds for both actors.

  Tracking uses ground truth for evaluation ONLY, never as student input. No
  mixed-policy rollout metric can open the takeover gate. Timeout is not failure.
  """
  env.seed(seed)
  obs, _ = env.reset()
  policy.eval()
  active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  sums = torch.zeros(env.num_envs, device=env.device)
  lengths = torch.zeros_like(sums)
  failures = torch.zeros_like(active)
  support_sums = torch.zeros_like(sums)
  has_support = False
  for _ in range(env.max_episode_length + 1):
    inputs = obs[teacher_group].clone() if teacher_group else student_input(obs)
    action = policy(inputs)
    if not torch.isfinite(action).all():
      raise FloatingPointError("non-finite validation action")
    raw = env.unwrapped
    robot = raw.scene["robot"]
    command = raw.command_manager.get_command("twist")
    # Evaluate the visited pre-step state, matching the command in obs.
    error_x = (robot.data.root_link_lin_vel_b[:, 0] - command[:, 0]).square()
    error_yaw = (robot.data.root_link_ang_vel_b[:, 2] - command[:, 2]).square()
    score = 0.5 * (torch.exp(-error_x / 0.25) + torch.exp(-error_yaw / 0.25))
    sums += score * active
    support = support_evaluation(raw)
    if support is not None:
      has_support = True
      support_sums += support * active
    lengths += active
    obs, _, done, extras = env.step(action)
    ended = active & done.bool()
    timeout = extras.get("time_outs")
    if timeout is None:
      raise ValueError("validation requires timeout provenance")
    # A failure on the timeout step is still a failure.
    failures |= ended & raw.reset_terminated.bool()
    active &= ~ended
    if not active.any():
      break
  if active.any():
    raise RuntimeError("validation horizon ended before all first episodes completed")
  return ValidationResult(
    float((sums / lengths.clamp_min(1)).mean()),
    float(failures.float().mean()),
    env.num_envs,
    float((support_sums / lengths.clamp_min(1)).mean()) if has_support else None,
  )


def load_teacher(manifest, checkpoint, contract, task, device):
  verify_teacher_manifest(manifest, checkpoint, contract, task)
  if task in ("wolf-methods-soft", "wolf-methods-hard"):
    if (
      manifest["input_group"] != "actor"
      or manifest["history_length"] != 6
      or manifest["history_order"] != "newest_first"
    ):
      raise ValueError("Methods teachers require the canonical actor history contract")
  if manifest.get("target_runtime_sha256") != runtime_digest():
    raise ValueError("Wolf runtime changed since teacher qualification")
  model = torch.jit.load(str(checkpoint), map_location=device).eval()
  source = ControlContract.from_dict(manifest["source_control"])
  return FrozenTeacher(
    model,
    TargetSpaceAdapter(source, contract),
    torch.tensor(manifest["observation_matrix"], dtype=torch.float32),
    torch.tensor(manifest["observation_bias"], dtype=torch.float32),
    history_length=manifest["history_length"],
    newest_first=manifest["history_order"] == "newest_first",
  ).to(device)


def save_checkpoint(path: Path, trainer, manifest, task, teacher_hash):
  state = trainer.state_dict()
  state.update(
    teacher_sha256=teacher_hash,
    teacher_manifest=manifest,
    task=task,
    target_runtime_sha256=runtime_digest(),
    target_control=asdict(current_contract(make_config(task, 1, trainer.cfg.seed))),
    curriculum_policy="frozen_initial_config",
    student_input_contract="actor only; frame-major newest-to-oldest; no normalizer",
    student_input_dim=trainer.student.input_dim,
    student_hidden_dims=trainer.student.hidden_dims,
    resume_semantics="optimizer/replay/RNG restored; simulator episodes restart",
  )
  temporary = path.with_suffix(".tmp")
  torch.save(state, temporary)
  temporary.replace(path)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="wolf-flat")
  parser.add_argument("--teacher", type=Path)
  parser.add_argument("--manifest", type=Path)
  parser.add_argument("--config", type=Path, help="JSON DaggerConfig overrides")
  parser.add_argument("--output", type=Path, default=Path("logs/wolf_distillation"))
  parser.add_argument("--resume", type=Path)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--validation-envs", type=int, default=16)
  parser.add_argument("--validation-seed", type=int, default=900_101)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--describe-contract", action="store_true")
  parser.add_argument("--train", action="store_true", help="Explicitly enable training")
  args = parser.parse_args()
  cfg = DaggerConfig(**(json.loads(args.config.read_text()) if args.config else {}))
  env_cfg = make_config(args.task, args.num_envs, cfg.seed)
  contract = current_contract(env_cfg)
  if args.describe_contract:
    print(
      json.dumps(
        {
          "target_task": args.task,
          "source_control": asdict(contract),
          "target_contract_sha256": contract.digest(),
          "target_runtime_sha256": runtime_digest(),
          "qualification": "unqualified",
          "note": "This describes the current contract, not a qualified teacher.",
        },
        indent=2,
      )
    )
    return
  if args.num_envs < 1 or args.validation_envs < 1:
    raise ValueError("environment counts must be positive")
  if args.validation_seed == cfg.seed:
    raise ValueError("training and validation seeds must differ")
  if args.teacher is None or args.manifest is None:
    parser.error("--teacher and --manifest are required")
  if args.train:
    cfg.validate_formal_run()
  manifest = json.loads(args.manifest.read_text())
  teacher = load_teacher(manifest, args.teacher, contract, args.task, args.device)
  torch.manual_seed(cfg.seed)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=args.device))
  validation_env = None
  try:
    obs, _ = env.reset()
    student = DeploymentStudent(student_input(obs).shape[-1]).to(args.device)
    trainer = DaggerTrainer(student, teacher, cfg, manifest["input_group"])
    teacher_hash = sha256_file(args.teacher)
    if args.resume:
      state = torch.load(args.resume, map_location="cpu", weights_only=True)
      if (
        state.get("teacher_sha256") != teacher_hash
        or state.get("teacher_manifest") != manifest
        or state.get("task") != args.task
        or state.get("target_runtime_sha256") != runtime_digest()
      ):
        raise ValueError("resume teacher/task/runtime mismatch")
      trainer.load_state_dict(state)
    if not args.train:
      before = {k: v.clone() for k, v in student.state_dict().items()}
      obs, report = trainer.collect(env, obs, steps=4)
      assert all(torch.equal(v, before[k]) for k, v in student.state_dict().items())
      print(json.dumps({"preflight_only": True, "optimizer_steps": 0, **report}))
      return
    # Fresh directory prevents accidental overwrite of a prior experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (args.output / "teacher_manifest.json").write_text(json.dumps(manifest, indent=2))
    validation_cfg = make_config(args.task, args.validation_envs, args.validation_seed)
    # Validation curricula must not change the benchmark across policy evaluations.
    validation_cfg.curriculum = {}
    validation_env = RslRlVecEnvWrapper(
      ManagerBasedRlEnv(validation_cfg, device=args.device)
    )
    reference = evaluate(
      validation_env, trainer.teacher, args.validation_seed, manifest["input_group"]
    )
    # Check reference eligibility even while takeover is disabled for BC ablation.
    if (
      reference.tracking < cfg.minimum_teacher_tracking
      or reference.failure_rate > cfg.maximum_failure_rate
      or (
        reference.support_coverage is not None
        and reference.support_coverage < cfg.minimum_teacher_support
      )
    ):
      raise ValueError("teacher failed the current validation qualification floor")
    log = args.output / "metrics.jsonl"
    while trainer.iteration < cfg.max_iterations:
      obs, collection_metrics = trainer.collect(env, obs)
      metrics = cast(dict[str, Any], collection_metrics)
      metrics.update(trainer.update())
      if trainer.iteration % cfg.eval_interval == 0:
        result = evaluate(validation_env, student, args.validation_seed)
        trainer.gate.observe(result, reference)
        metrics.update(validation=asdict(result), teacher_validation=asdict(reference))
        save_checkpoint(
          args.output / f"student_{trainer.iteration}.pt",
          trainer,
          manifest,
          args.task,
          teacher_hash,
        )
      metrics.update(
        train_iter=trainer.iteration,
        optimizer_steps=trainer.optimizer_steps,
        transitions=trainer.transitions,
        replay_samples=trainer.replay.size,
        student_probability=trainer.gate.student_probability,
      )
      with log.open("a") as stream:
        stream.write(json.dumps(metrics) + "\n")
      print(json.dumps(metrics), flush=True)
    save_checkpoint(
      args.output / "student_final.pt", trainer, manifest, args.task, teacher_hash
    )
    torch.jit.script(student.cpu().eval()).save(str(args.output / "student_actor.pt"))
  finally:
    env.close()
    if validation_env is not None:
      validation_env.close()


if __name__ == "__main__":
  main()
