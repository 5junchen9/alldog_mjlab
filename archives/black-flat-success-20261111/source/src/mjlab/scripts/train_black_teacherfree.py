"""Single-policy, teacher-free continuation from the selected +1000 checkpoint."""

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
from mjlab.rl.black_teacherfree import TaskLoggedPPO
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.scripts.black_seed_matrix import ROOT, atomic_json, file_sha
from mjlab.tasks.velocity.config.black_teacher.rl_cfg import (
  black_teacher_ppo_runner_cfg,
)
from mjlab.tasks.velocity.config.black_teacher.teacherfree_command import (
  TeacherFreeVelocityCommand,
  TeacherFreeVelocityCommandCfg,
)
from mjlab.tasks.velocity.config.black_teacher.teacherfree_env_cfg import (
  black_teacherfree_env_cfg,
)
from mjlab.tasks.velocity.config.black_teacher.teacherfree_rewards import (
  command_gates,
  configure_teacherfree_rewards,
)
from mjlab.utils.os import dump_yaml

SOURCE = (
  ROOT
  / "logs/black_continuation/omni_chain_20261110/stage1/updates_01000/model_3999.pt"
)
SOURCE_SHA = "ea4a072fe6a11eb1fb29cfc4e2c56babfb2fad5a5f2ebe1778868389e792a1fb"
START_ITERATION = 3999


def equal_tree(a, b):
  if isinstance(a, torch.Tensor):
    return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
  if isinstance(a, dict):
    return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
  if isinstance(a, (tuple, list)):
    return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b, strict=True))
  return a == b


def make_configs(num_envs, seed):
  cfg = black_teacherfree_env_cfg()
  configure_teacherfree_rewards(cfg)
  rl = black_teacher_ppo_runner_cfg()
  cfg.seed = rl.seed = seed
  cfg.scene.num_envs = num_envs
  assert isinstance(cfg.commands["twist"], TeacherFreeVelocityCommandCfg)
  cfg.commands["twist"].debug_vis = False
  cfg.commands["twist"].profile_seed = seed
  rl.algorithm.class_name = "mjlab.rl.black_teacherfree:TaskLoggedPPO"
  rl.algorithm.learning_rate = 5e-5
  rl.algorithm.schedule = "fixed"
  rl.save_interval = 100
  rl.logger, rl.upload_model = "tensorboard", False
  return cfg, rl


class ProfileLoggedWrapper(RslRlVecEnvWrapper):
  def step(self, actions):
    # Reward uses this applied command before resampling/reset in env.step().
    command = self.unwrapped.command_manager.get_term("twist")
    assert isinstance(command, TeacherFreeVelocityCommand)
    before = command.command.clone()
    low_gate, high_gate = command_gates(before)
    out = super().step(actions)
    rates = self.unwrapped.reward_manager._step_reward
    logs = out[3]["log"]
    for index, name in enumerate(("low", "omni", "high")):
      selected = command.profile_id == index
      if not selected.any():
        continue
      logs[f"TaskReward/{name}/total_rate"] = (
        out[1][selected] / self.unwrapped.step_dt
      ).mean()
      logs[f"TaskReward/{name}/low_gate"] = low_gate[selected].mean()
      logs[f"TaskReward/{name}/high_gate"] = high_gate[selected].mean()
      logs[f"TaskReward/{name}/terminated_fraction"] = (
        self.unwrapped.reset_terminated[selected].float().mean()
      )
      for j, term in enumerate(self.unwrapped.reward_manager.active_terms):
        logs[f"TaskReward/{name}/{term}"] = rates[selected, j].mean()
    if self.unwrapped.common_step_counter % 100 == 0:
      torch.testing.assert_close(
        rates.sum(-1) * self.unwrapped.step_dt, out[1], atol=1e-6, rtol=1e-5
      )
    return out


def train(args):
  if importlib.metadata.version("rsl-rl-lib") != "5.5.1":
    raise RuntimeError("Require verified RSL-RL 5.5.1")
  source, output = args.source.resolve(), args.output.resolve()
  if output.exists():
    raise FileExistsError("Fresh output directory required")
  if file_sha(SOURCE) != SOURCE_SHA:
    raise RuntimeError("Protected +1000 source changed")
  payload = torch.load(source, map_location="cpu", weights_only=False)
  parent_sha = file_sha(source)
  if args.offset == 0:
    if source != SOURCE.resolve() or parent_sha != SOURCE_SHA:
      raise ValueError("Must start from selected +1000 model3999")
  else:
    lineage = (payload.get("infos") or {}).get("teacherfree", {})
    if (
      lineage.get("completed_updates") != args.offset
      or lineage.get("initial_source_sha256") != SOURCE_SHA
    ):
      raise ValueError("Continuation provenance/count mismatch")
  if payload["iter"] != START_ITERATION + args.offset:
    raise ValueError("Global iteration mismatch")
  output.mkdir(parents=True)
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.set_num_threads(4)
  torch.set_num_interop_threads(2)
  cfg, rl = make_configs(args.num_envs, args.seed)
  dump_yaml(output / "params/env.yaml", asdict(cfg))
  dump_yaml(output / "params/agent.yaml", asdict(rl))
  lineage = dict(
    source=str(source),
    source_sha256=parent_sha,
    initial_source=str(SOURCE),
    initial_source_sha256=SOURCE_SHA,
    offset=args.offset,
    requested_updates=args.updates,
    seed=args.seed,
    num_envs=args.num_envs,
    actor_count=1,
    critic_count=1,
    teachers_enabled=False,
    dagger_enabled=False,
    him_enabled=False,
    reward_condition="Current applied command, never hidden profile ID",
    learning_rate=5e-5,
    schedule="fixed",
    scope="Actor/critic/normalizers/Adam/common counter restored; fresh simulator episodes and seeded RNG",
  )
  status_path = output / "status.json"
  atomic_json(status_path, dict(state="running", pid=os.getpid(), lineage=lineage))
  stop = threading.Event()
  samples = []

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
        samples.append(dict(time=time.time(), gpu=gpu))
        atomic_json(output / "gpu_samples.json", samples)
      except Exception as error:
        samples.append(dict(time=time.time(), error=str(error)))

  thread = threading.Thread(target=monitor, daemon=True)
  if args.device.startswith("cuda"):
    thread.start()

  class AtomicRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
      for module in (self.alg.actor, self.alg.critic):
        if any(not torch.isfinite(v).all() for v in module.state_dict().values()):
          raise FloatingPointError("Nonfinite actor/critic/normalizers")
      target = Path(path)
      if target.exists():
        return
      completed = self.current_learning_iteration - START_ITERATION
      temporary = target.with_name(f".{target.name}.partial")
      super().save(
        str(temporary),
        {**(infos or {}), "teacherfree": {**lineage, "completed_updates": completed}},
      )
      os.replace(temporary, target)
      record = dict(
        path=str(target),
        sha256=file_sha(target),
        global_iteration=self.current_learning_iteration,
        completed_updates=completed,
      )
      atomic_json(target.with_suffix(".pt.json"), record)
      atomic_json(
        status_path,
        dict(
          state="running",
          pid=os.getpid(),
          checkpoint=record,
          completed_updates=completed,
          lineage=lineage,
        ),
      )

  env = None
  started = time.time()
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    wrapper = ProfileLoggedWrapper(env, clip_actions=rl.clip_actions)
    runner = AtomicRunner(wrapper, asdict(rl), str(output), args.device)
    assert isinstance(runner.alg, TaskLoggedPPO)
    runner.load(str(source), map_location=args.device)
    verified = dict(
      actor_exact=equal_tree(
        payload["actor_state_dict"], runner.alg.actor.state_dict()
      ),
      critic_exact=equal_tree(
        payload["critic_state_dict"], runner.alg.critic.state_dict()
      ),
      adam_moments_exact=equal_tree(
        payload["optimizer_state_dict"]["state"],
        runner.alg.optimizer.state_dict()["state"],
      ),
      env_counter_restored=env.common_step_counter,
      first_global_iteration=payload["iter"] + 1,
      prior_dagger_not_loaded=True,
      teacher_modules_count=0,
    )
    if not all(
      verified[k] for k in ("actor_exact", "critic_exact", "adam_moments_exact")
    ):
      raise AssertionError("Full-state load verification failed")
    runner.alg.learning_rate = 5e-5
    runner.alg.schedule = "fixed"
    for group in runner.alg.optimizer.param_groups:
      group["lr"] = 5e-5
    command = env.command_manager.get_term("twist")
    assert isinstance(command, TeacherFreeVelocityCommand)
    runner.alg.profile_id = command.profile_id
    verified["profile_counts"] = torch.bincount(
      command.profile_id, minlength=3
    ).tolist()
    verified["explicit_learning_rate"] = 5e-5
    atomic_json(output / "load_verified.json", {**verified, "lineage": lineage})
    runner.current_learning_iteration = payload["iter"] + 1
    del payload
    runner.learn(args.updates, init_at_random_ep_len=False)
    expected = START_ITERATION + args.offset + args.updates
    if runner.current_learning_iteration != expected:
      raise AssertionError("Incorrect final update index")
    final = output / f"model_{expected}.pt"
    record = json.loads(final.with_suffix(".pt.json").read_text())
    if file_sha(final) != record["sha256"]:
      raise AssertionError("Final checkpoint hash mismatch")
    atomic_json(
      status_path,
      dict(
        state="completed",
        checkpoint=record,
        completed_updates=args.offset + args.updates,
        global_iteration=expected,
        lineage=lineage,
        elapsed_s=time.time() - started,
        torch_peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20
        if args.device.startswith("cuda")
        else None,
      ),
    )
    print("TEACHERFREE_CHUNK_COMPLETE", json.dumps(record), flush=True)
  except BaseException as error:
    atomic_json(
      status_path,
      dict(state="failed", error=f"{type(error).__name__}: {error}", lineage=lineage),
    )
    raise
  finally:
    stop.set()
    if thread.is_alive():
      thread.join(10)
    atomic_json(output / "gpu_samples.json", samples)
    if env is not None:
      env.close()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--source", type=Path, required=True)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--offset", type=int, default=0)
  p.add_argument("--updates", type=int, required=True)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--seed", type=int, default=51001)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--execute", action="store_true")
  args = p.parse_args()
  if args.offset < 0 or args.updates <= 0 or args.offset + args.updates > 30000:
    raise ValueError("Single continuation budget is at most30000 additional updates")
  if args.execute:
    train(args)
  else:
    print("Dryrun; use --execute", vars(args))


if __name__ == "__main__":
  main()
