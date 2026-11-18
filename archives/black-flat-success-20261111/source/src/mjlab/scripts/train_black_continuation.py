"""One full-state continuation chunk. Never modifies a source checkpoint."""

import argparse
import importlib.metadata
import json
import os
import random
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.black_continuation import SelectiveTeacherPPO
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.scripts.black_seed_matrix import ROOT, atomic_json, file_sha
from mjlab.scripts.evaluate_black_candidates import BASELINE, BASELINE_SHA
from mjlab.scripts.evaluate_black_continuation import _verified
from mjlab.tasks.velocity.config.black_teacher.continuation_env_cfg import (
  black_continuation_env_cfg,
)
from mjlab.tasks.velocity.config.black_teacher.rl_cfg import (
  black_teacher_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml

LOW = (
  ROOT
  / "logs/black_seed_matrix/black_low_high_candidates_20261110/low_fine/seed_31001/attempts/1794314750570609651/model_2999.pt"
)
LOW_SHA = "df9ed9f241ce513a22945bc73d2a6baa7c8142e8ed46b5361f49daa17993b593"


def equal_tree(a, b):
  if isinstance(a, torch.Tensor):
    return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
  if isinstance(a, dict):
    return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
  if isinstance(a, (tuple, list)):
    return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b, strict=True))
  return a == b


def train(args):
  if importlib.metadata.version("rsl-rl-lib") != "5.5.1":
    raise RuntimeError("PPO adapter requires rsl-rl-lib 5.5.1")
  source, output = args.source.resolve(), args.output.resolve()
  if output.exists():
    raise FileExistsError("Fresh chunk directory required; source never overwritten")
  if file_sha(BASELINE) != BASELINE_SHA or file_sha(LOW) != LOW_SHA:
    raise RuntimeError("Frozen teacher hash changed")
  payload = torch.load(source, weights_only=False, map_location="cpu")
  parent_sha = file_sha(source)
  parent_iteration = payload["iter"]
  continuation = payload.get("continuation")
  if args.stage == 1 and args.stage_offset == 0:
    if source != BASELINE.resolve() or parent_sha != BASELINE_SHA:
      raise ValueError("Stage1 must start from the original Omni checkpoint")
  elif args.stage == 2 and args.stage_offset == 0:
    if not continuation or continuation["stage"] != 1 or args.parent_validation is None:
      raise ValueError("Stage2 requires a validated stage1 checkpoint")
    report = json.loads(args.parent_validation.read_text())
    if not _verified(report) or report["checkpoint"]["sha256"] != parent_sha:
      raise ValueError("Stage1 parent validation mismatch")
  elif (
    not continuation
    or continuation["stage"] != args.stage
    or continuation["stage_update"] != args.stage_offset
  ):
    raise ValueError("Continuation stage/update lineage mismatch")
  output.mkdir(parents=True)
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.set_num_threads(4)
  torch.set_num_interop_threads(2)
  cfg = black_continuation_env_cfg(f"stage{args.stage}")
  rl = black_teacher_ppo_runner_cfg()
  cfg.seed = rl.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.commands["twist"].debug_vis = False
  rl.algorithm.class_name = "mjlab.rl.black_continuation:SelectiveTeacherPPO"
  rl.algorithm.learning_rate = 1e-4
  rl.save_interval = 100
  rl.max_iterations = args.updates
  rl.logger, rl.upload_model = "tensorboard", False
  dump_yaml(output / "params" / "env.yaml", asdict(cfg))
  dump_yaml(output / "params" / "agent.yaml", asdict(rl))
  lineage = dict(
    stage=args.stage,
    stage_offset=args.stage_offset,
    source=str(source),
    source_sha256=parent_sha,
    source_iteration=parent_iteration,
    baseline=str(BASELINE),
    baseline_sha256=BASELINE_SHA,
    low_teacher=str(LOW),
    low_teacher_sha256=LOW_SHA,
    parent_validation=str(args.parent_validation) if args.parent_validation else None,
    seed=args.seed,
    num_envs=args.num_envs,
    requested_updates=args.updates,
    scope="Full actor/critic/Adam/normalizer/DAgger state; new simulator episodes and RNG, not bitwise trajectories",
    high_teacher_enabled=False,
    him_enabled=False,
  )
  state_path = output / "status.json"
  atomic_json(state_path, dict(state="running", pid=os.getpid(), lineage=lineage))
  gpu_samples = []
  stop = threading.Event()

  def monitor():
    while not stop.wait(10):
      try:
        gpu = subprocess.check_output(
          [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
          ],
          text=True,
          timeout=8,
        ).strip()
        gpu_samples.append(dict(time=time.time(), gpu=gpu))
        atomic_json(output / "gpu_samples.json", gpu_samples)
      except Exception as error:
        gpu_samples.append(dict(time=time.time(), error=str(error)))

  thread = threading.Thread(target=monitor, daemon=True)
  if args.device.startswith("cuda"):
    thread.start()

  class AtomicRunner(MjlabOnPolicyRunner):
    @property
    def continuation(self) -> SelectiveTeacherPPO:
      assert isinstance(self.alg, SelectiveTeacherPPO)
      return self.alg

    def save(self, path, infos=None):
      for module in (self.alg.actor, self.alg.critic):
        if any(not torch.isfinite(v).all() for v in module.state_dict().values()):
          raise FloatingPointError("Nonfinite policy/normalizer; no checkpoint")
      target = Path(path)
      if target.exists():
        return  # RSL may save the same last iteration twice; never overwrite.
      temporary = target.with_name(f".{target.name}.partial")
      super().save(str(temporary), {**(infos or {}), "black_continuation": lineage})
      os.replace(temporary, target)
      record = dict(
        path=str(target),
        sha256=file_sha(target),
        global_iteration=self.current_learning_iteration,
        completed_stage_updates=self.continuation.stage_update,
      )
      atomic_json(target.with_suffix(".pt.json"), record)
      atomic_json(
        state_path,
        dict(
          state="running",
          pid=os.getpid(),
          lineage=lineage,
          checkpoint=record,
          completed_stage_updates=self.continuation.stage_update,
        ),
      )

  env = None
  started = time.time()
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
    runner = AtomicRunner(wrapper, asdict(rl), str(output), args.device)
    runner.load(str(source), map_location=args.device)
    verified = {
      "actor_exact": equal_tree(
        payload["actor_state_dict"], runner.alg.actor.state_dict()
      ),
      "critic_exact": equal_tree(
        payload["critic_state_dict"], runner.alg.critic.state_dict()
      ),
      "adam_moments_exact": equal_tree(
        payload["optimizer_state_dict"]["state"],
        runner.alg.optimizer.state_dict()["state"],
      ),
      "learning_rate_after_explicit_cap": runner.alg.learning_rate,
      "source_iteration": parent_iteration,
      "first_update_iteration": parent_iteration + 1,
      "env_counter_restored": env.common_step_counter,
    }
    if not all(
      verified[k] for k in ("actor_exact", "critic_exact", "adam_moments_exact")
    ):
      raise AssertionError(
        "Warm start must preserve exact weights, normalizers and Adam moments"
      )
    runner.current_learning_iteration = parent_iteration + 1
    runner.continuation.bind(env, str(LOW), str(BASELINE), args.stage)
    runner.continuation.stage_update = args.stage_offset
    teacher_before = [
      {k: v.detach().clone() for k, v in t.state_dict().items()}
      for t in runner.continuation.teachers
    ]
    atomic_json(output / "load_verified.json", {**verified, "lineage": lineage})
    del payload
    runner.learn(args.updates, init_at_random_ep_len=False)
    for before, teacher in zip(
      teacher_before, runner.continuation.teachers, strict=True
    ):
      if not equal_tree(before, teacher.state_dict()) or any(
        p.grad is not None for p in teacher.parameters()
      ):
        raise AssertionError("Frozen teacher or normalization changed")
    expected = parent_iteration + args.updates
    final = output / f"model_{expected}.pt"
    if (
      runner.current_learning_iteration != expected
      or runner.continuation.stage_update != args.stage_offset + args.updates
    ):
      raise AssertionError("Incorrect update count")
    record = json.loads(final.with_suffix(".pt.json").read_text())
    if file_sha(final) != record["sha256"]:
      raise AssertionError("Checkpoint hash mismatch")
    atomic_json(
      state_path,
      dict(
        state="completed",
        checkpoint=record,
        completed_stage_updates=runner.continuation.stage_update,
        lineage=lineage,
        elapsed_s=time.time() - started,
        teachers_unchanged=True,
        torch_peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20
        if args.device.startswith("cuda")
        else None,
      ),
    )
    print("CONTINUATION_CHUNK_COMPLETE", json.dumps(record), flush=True)
  except BaseException as error:
    atomic_json(
      state_path,
      dict(state="failed", error=f"{type(error).__name__}: {error}", lineage=lineage),
    )
    raise
  finally:
    stop.set()
    if thread.is_alive():
      thread.join(10)
    atomic_json(output / "gpu_samples.json", gpu_samples)
    if env is not None:
      env.close()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--source", type=Path, required=True)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--stage", type=int, choices=(1, 2), required=True)
  p.add_argument("--stage-offset", type=int, default=0)
  p.add_argument("--updates", type=int, required=True)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--seed", type=int, default=41001)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--parent-validation", type=Path)
  p.add_argument("--execute", action="store_true")
  args = p.parse_args()
  if (
    args.stage_offset < 0
    or args.updates <= 0
    or args.stage_offset + args.updates > 10000
  ):
    raise ValueError("Each stage is capped at 10000 additional updates")
  if args.execute:
    train(args)
  else:
    print("Dry run; use --execute to run", vars(args))


if __name__ == "__main__":
  main()
