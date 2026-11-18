"""Serial two-stage Black continuation, with independent validation and rollback.

Default is a side-effect-free dry-run. Final holdout never selects checkpoints.
"""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mjlab.scripts.black_seed_matrix import (
  ROOT,
  atomic_json,
  digest,
  file_sha,
  guard_resources,
  lock,
  read_json,
  source_files,
  source_manifest,
)

SOURCE = ROOT / (
  "logs/rsl_rl/black_teacher_omni_flat/2026-11-10_19-06-42_omni_4096_3000/model_2999.pt"
)


def chunk_schedule(iterations: int, chunk_updates: int) -> list[tuple[int, int]]:
  if not 1 <= iterations <= 10000 or not 1 <= chunk_updates <= 1000:
    raise ValueError("Budget must be 1..10000 and chunk updates 1..1000")
  return [
    (o, min(chunk_updates, iterations - o)) for o in range(0, iterations, chunk_updates)
  ]


def checkpoint_record(path: Path) -> dict[str, str]:
  return dict(path=str(path.resolve(strict=True)), sha256=file_sha(path))


def verify_checkpoint(record: dict[str, Any]) -> Path:
  path = Path(record["path"]).resolve(strict=True)
  if file_sha(path) != record["sha256"]:
    raise RuntimeError(f"Checkpoint changed: {path}")
  return path


def verify_sources(sources: dict[str, str]) -> None:
  paths = source_files()
  if {str(p.relative_to(ROOT)) for p in paths} != set(sources):
    raise RuntimeError("Source file set changed after freeze")
  for path in paths:
    if file_sha(path) != sources[str(path.relative_to(ROOT))]:
      raise RuntimeError(f"Source changed after freeze: {path}")


def make_plan(output: Path, iterations: int, chunk_updates: int) -> dict[str, Any]:
  chunks = chunk_schedule(iterations, chunk_updates)
  return dict(
    schema_version=1,
    output=str(output.resolve()),
    source=str(SOURCE),
    iterations_per_stage=iterations,
    chunk_updates=chunk_updates,
    chunks=[dict(offset=o, updates=n) for o, n in chunks],
    stages=[dict(stage=1, seed=41001), dict(stage=2, seed=42001)],
    num_envs=4096,
    device="cuda:0",
    save_interval=100,
    parallel_jobs=1,
    min_free_gpu_mb=6000,
    min_free_disk_gb=15,
    resume_scope="Restore actor, critic, optimizer, normalization and common step "
    "counter. Simulator episodes and RNG streams restart; not bitwise resume.",
    selection="Latest validated eligible checkpoint before first stop. Severe or "
    "repeated independent validation deterioration stops stage. Stage 1 stop "
    "prohibits Stage 2. Final holdout is report-only.",
  )


class Supervisor:
  def __init__(self, plan: dict[str, Any], fd: int):
    self.plan, self.fd = plan, fd
    self.output = Path(plan["output"])
    self.state: dict[str, Any] = dict(
      state="active",
      supervisor_pid=os.getpid(),
      child_pid=None,
      accepted_checkpoint=plan["source_checkpoint"],
      stages=[],
      started_at=time.time(),
      fingerprint=plan["fingerprint"],
    )
    self.save()

  def save(self) -> None:
    self.state["updated_at"] = time.time()
    atomic_json(self.output / "status.json", self.state)

  def run_child(self, command: list[str], label: str, seed: int) -> None:
    verify_sources(self.plan["sources"])
    verify_checkpoint(self.plan["source_checkpoint"])
    guard_resources(self.plan, self.output)
    env = os.environ.copy()
    for key in (
      "RANK",
      "LOCAL_RANK",
      "WORLD_SIZE",
      "LOCAL_WORLD_SIZE",
      "MASTER_ADDR",
      "MASTER_PORT",
      "GROUP_RANK",
      "ROLE_RANK",
    ):
      env.pop(key, None)
    env.update(
      PYTHONHASHSEED=str(seed),
      OMP_NUM_THREADS="4",
      OPENBLAS_NUM_THREADS="2",
      MKL_NUM_THREADS="2",
    )
    with (self.output / "console.log").open("a", buffering=1) as stream:
      stream.write(f"\nSTART {label} {time.time()} {command!r}\n")
      proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        pass_fds=(self.fd,),
      )
      self.state.update(
        child_pid=proc.pid,
        child_label=label,
        child_command=command,
        child_started_at=time.time(),
      )
      self.save()

      def stop(signum, _frame):
        with contextlib.suppress(ProcessLookupError):
          os.killpg(proc.pid, signum)
        raise InterruptedError(f"Supervisor received signal {signum}")

      old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
      try:
        code = proc.wait()
      finally:
        if proc.poll() is None:
          with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
          try:
            proc.wait(timeout=60)
          except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
              os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        for sig, handler in old.items():
          signal.signal(sig, handler)
        self.state.update(child_pid=None, child_exit_code=proc.returncode)
        self.save()
      stream.write(f"END {label} {time.time()} exit={code}\n")
    if code:
      raise RuntimeError(f"{label} exited {code}; see console.log")
    verify_sources(self.plan["sources"])
    verify_checkpoint(self.plan["source_checkpoint"])

  def evaluate(
    self, record: dict[str, Any], output: Path, split: str = "validation"
  ) -> dict[str, Any]:
    checkpoint = verify_checkpoint(record)
    self.run_child(
      [
        sys.executable,
        "-m",
        "mjlab.scripts.evaluate_black_continuation",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        "cuda:0",
        "--split",
        split,
        "--execute",
      ],
      f"evaluate_{output.stem}",
      921101 if split == "validation" else 931101,
    )
    report = read_json(output)
    if (
      not report.get("complete")
      or not report.get("checkpoint_unchanged")
      or report.get("checkpoint") != checkpoint_record(checkpoint)
    ):
      raise RuntimeError("Evaluator returned incomplete or mismatched checkpoint")
    return report

  def execute(self) -> None:
    # Lazy import keeps planning free of simulator/GPU module initialization.
    from mjlab.scripts.evaluate_black_continuation import gate_checkpoint

    baseline_path = self.output / "baseline_validation.json"
    reference = self.evaluate(self.plan["source_checkpoint"], baseline_path)
    if gate_checkpoint(reference, reference)["stop"]:
      raise RuntimeError("Baseline validation is not eligible")
    accepted = self.plan["source_checkpoint"]
    accepted_path = baseline_path
    stage1_reference = None
    for stage in (1, 2):
      row: dict[str, Any] = dict(
        stage=stage,
        state="active",
        started_at=time.time(),
        source_checkpoint=accepted,
        completed_updates=0,
        accepted_checkpoint=accepted,
        validations=[],
      )
      self.state["stages"].append(row)
      self.state["stage"] = stage
      self.save()
      previous = None
      for chunk in self.plan["chunks"]:
        offset, updates = chunk["offset"], chunk["updates"]
        child_output = self.output / f"stage{stage}" / f"updates_{offset + updates:05d}"
        command = [
          sys.executable,
          "-m",
          "mjlab.scripts.train_black_continuation",
          "--source",
          str(verify_checkpoint(accepted)),
          "--output",
          str(child_output),
          "--stage",
          str(stage),
          "--stage-offset",
          str(offset),
          "--updates",
          str(updates),
          "--num-envs",
          "4096",
          "--seed",
          str(41001 if stage == 1 else 42001),
          "--device",
          "cuda:0",
          "--execute",
        ]
        if stage == 2 and offset == 0:
          command += ["--parent-validation", str(accepted_path)]
        self.run_child(
          command,
          f"train_stage{stage}_{offset + updates}",
          41001 if stage == 1 else 42001,
        )
        status = read_json(child_output / "status.json")
        if (
          status.get("state") != "completed"
          or status.get("completed_stage_updates") != offset + updates
        ):
          raise RuntimeError("Train child did not finish requested stage updates")
        candidate = status["checkpoint"]
        verify_checkpoint(candidate).relative_to(child_output.resolve())
        validation_path = child_output / "validation.json"
        current = self.evaluate(candidate, validation_path)
        decision = gate_checkpoint(
          reference, current, previous, stage1_reference=stage1_reference
        )
        row["completed_updates"] = offset + updates
        row["validations"].append(
          dict(report=str(validation_path), decision=decision, checkpoint=candidate)
        )
        if decision["stop"]:
          row.update(
            state="stopped_safely",
            stop_reasons=decision["reasons"],
            rollback_checkpoint=accepted,
          )
          self.save()
          break
        if not decision["eligible"]:
          raise RuntimeError("Gate returned neither eligible nor stop")
        accepted, accepted_path = candidate, validation_path
        row["accepted_checkpoint"] = accepted
        self.state["accepted_checkpoint"] = accepted
        previous = current
        self.save()
      else:
        row["state"] = "completed"
      row["finished_at"] = time.time()
      self.save()
      if row["state"] != "completed":
        break
      if stage == 1:
        stage1_reference = read_json(accepted_path)
        self.state["selected_stage1_checkpoint"] = accepted
        self.state["selected_stage1_validation"] = str(accepted_path)
        self.save()
    # Freeze selection before reading held-out command/seed scores.
    self.state.update(
      selected_checkpoint=accepted,
      selected_validation=str(accepted_path),
      selection_finished_at=time.time(),
    )
    self.save()
    final_path = self.output / "final_holdout.json"
    self.evaluate(accepted, final_path, "final")
    self.state.update(
      state="completed"
      if all(r["state"] == "completed" for r in self.state["stages"])
      else "stopped_safely",
      final_holdout=str(final_path),
      finished_at=time.time(),
    )
    self.save()


def execute_plan(plan: dict[str, Any]) -> None:
  output = Path(plan["output"])
  global_lock = Path(f"/tmp/mjlab_black_matrix_{os.getuid()}_cuda_0.lock")
  with lock(global_lock) as fd:
    if output.exists():
      raise FileExistsError(f"Refusing existing output directory: {output}")
    guard_resources(plan, output)
    frozen = {
      **plan,
      "sources": source_manifest(),
      "source_checkpoint": checkpoint_record(SOURCE),
    }
    frozen["fingerprint"] = digest(frozen)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", frozen)
    supervisor = Supervisor(frozen, fd)
    try:
      supervisor.execute()
    except BaseException as error:
      supervisor.state.update(
        state="interrupted"
        if isinstance(error, (KeyboardInterrupt, InterruptedError))
        else "failed",
        error=str(error),
        finished_at=time.time(),
      )
      supervisor.save()
      raise


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--iterations", type=int, default=10000)
  parser.add_argument("--chunk-updates", type=int, default=1000)
  parser.add_argument("--execute", action="store_true")
  args = parser.parse_args()
  plan = make_plan(args.output, args.iterations, args.chunk_updates)
  if args.execute:
    execute_plan(plan)
  else:
    print(json.dumps(plan, indent=2))
    print(
      "DRY RUN: no files, training, evaluation, or GPU initialization", file=sys.stderr
    )


if __name__ == "__main__":
  main()
