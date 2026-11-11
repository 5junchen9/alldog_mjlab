"""One teacher-free lineage, at most 30000 additional updates; dry-run by default."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mjlab.scripts.black_seed_matrix import (
  ROOT,
  atomic_json,
  digest,
  guard_resources,
  lock,
  read_json,
  source_manifest,
)
from mjlab.scripts.run_black_continuation import (
  Supervisor as BaseSupervisor,
)
from mjlab.scripts.run_black_continuation import (
  checkpoint_record,
  verify_checkpoint,
)

BASELINE = ROOT / (
  "logs/rsl_rl/black_teacher_omni_flat/2026-11-10_19-06-42_omni_4096_3000/model_2999.pt"
)
SOURCE = ROOT / (
  "logs/black_continuation/omni_chain_20261110/stage1/updates_01000/model_3999.pt"
)
SOURCE_SHA256 = "ea4a072fe6a11eb1fb29cfc4e2c56babfb2fad5a5f2ebe1778868389e792a1fb"


def chunk_schedule(iterations: int, chunk_updates: int) -> list[tuple[int, int]]:
  if not 1 <= iterations <= 30000 or not 1 <= chunk_updates <= 1000:
    raise ValueError("Budget must be 1..30000 and chunk updates 1..1000")
  return [
    (o, min(chunk_updates, iterations - o)) for o in range(0, iterations, chunk_updates)
  ]


def make_plan(output: Path, iterations: int, chunk_updates: int) -> dict[str, Any]:
  return dict(
    schema_version=1,
    output=str(output.resolve()),
    source=str(SOURCE),
    source_sha256=SOURCE_SHA256,
    source_global_iteration=3999,
    baseline=str(BASELINE),
    additional_updates=iterations,
    chunk_updates=chunk_updates,
    chunks=[
      dict(offset=o, updates=n) for o, n in chunk_schedule(iterations, chunk_updates)
    ],
    training_seed_base=51001,
    num_envs=4096,
    device="cuda:0",
    save_interval=100,
    parallel_jobs=1,
    min_free_gpu_mb=6000,
    min_free_disk_gb=15,
    training="One actor, one critic; joint low/omni/high conditioned rewards. "
    "No teacher inference, HIM, or DAgger.",
    resume_scope="Restore actor, critic, optimizer, normalization and common step "
    "counter. Simulator episodes and RNG streams restart with distinct chunk seeds; "
    "not bitwise resume.",
    selection="Latest accepted checkpoint before first validation stop, or original "
    "source if none passed. No restart. Final paired holdout is report-only.",
  )


class Supervisor(BaseSupervisor):
  """Reuse frozen subprocess/lock inheritance, not the original staged workflow."""

  def __init__(self, plan: dict[str, Any], fd: int):
    self.plan, self.fd = plan, fd
    self.output = Path(plan["output"])
    self.state: dict[str, Any] = dict(
      state="active",
      supervisor_pid=os.getpid(),
      child_pid=None,
      accepted_checkpoint=plan["source_checkpoint"],
      completed_updates=0,
      global_iteration=plan["source_global_iteration"],
      validations=[],
      started_at=time.time(),
      fingerprint=plan["fingerprint"],
    )
    self.save()

  def run_child(self, command: list[str], label: str, seed: int) -> None:
    verify_checkpoint(self.plan["baseline_checkpoint"])
    super().run_child(command, label, seed)
    verify_checkpoint(self.plan["baseline_checkpoint"])

  def evaluate(
    self, record: dict[str, Any], output: Path, split: str = "validation"
  ) -> dict[str, Any]:
    from mjlab.scripts.evaluate_black_teacherfree import _verified

    checkpoint = verify_checkpoint(record)
    self.run_child(
      [
        sys.executable,
        "-m",
        "mjlab.scripts.evaluate_black_teacherfree",
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
      941101 if split == "validation" else 951101,
    )
    report = read_json(output)
    if not _verified(report, split=split) or report.get(
      "checkpoint"
    ) != checkpoint_record(checkpoint):
      raise RuntimeError("Evaluator returned incomplete or mismatched checkpoint")
    verify_checkpoint(record)
    return report

  def execute(self) -> None:
    from mjlab.scripts.evaluate_black_teacherfree import (
      _verified,
      gate_checkpoint,
      paired_final_report,
    )

    baseline_path = self.output / "baseline_validation.json"
    reference = self.evaluate(self.plan["baseline_checkpoint"], baseline_path)
    if gate_checkpoint(reference, reference)["stop"]:
      raise RuntimeError("Baseline validation is not eligible")
    source_path = self.output / "source_validation.json"
    source_reference = self.evaluate(self.plan["source_checkpoint"], source_path)
    if not _verified(source_reference):
      raise RuntimeError("Source validation is incomplete or unverified")
    # A weak starting score is recorded, not a reason to refuse repair training.
    self.state["source_gate"] = gate_checkpoint(reference, source_reference)
    accepted = self.plan["source_checkpoint"]
    accepted_path = source_path
    previous = None
    stopped = False
    self.save()
    for chunk in self.plan["chunks"]:
      offset, updates = chunk["offset"], chunk["updates"]
      child_output = self.output / f"updates_{offset + updates:05d}"
      seed = self.plan["training_seed_base"] + offset
      command = [
        sys.executable,
        "-m",
        "mjlab.scripts.train_black_teacherfree",
        "--source",
        str(verify_checkpoint(accepted)),
        "--output",
        str(child_output),
        "--offset",
        str(offset),
        "--updates",
        str(updates),
        "--num-envs",
        "4096",
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--execute",
      ]
      self.run_child(command, f"train_{offset + updates}", seed)
      status = read_json(child_output / "status.json")
      expected_iteration = self.plan["source_global_iteration"] + offset + updates
      if (
        status.get("state") != "completed"
        or status.get("completed_updates") != offset + updates
        or status.get("global_iteration") != expected_iteration
      ):
        raise RuntimeError("Train child did not finish requested cumulative updates")
      candidate = status["checkpoint"]
      verify_checkpoint(candidate).relative_to(child_output.resolve())
      self.state.update(
        completed_updates=offset + updates,
        global_iteration=expected_iteration,
        latest_checkpoint=candidate,
      )
      self.save()
      validation_path = child_output / "validation.json"
      current = self.evaluate(candidate, validation_path)
      decision = gate_checkpoint(
        reference, current, previous, stage1_reference=source_reference
      )
      self.state["validations"].append(
        dict(report=str(validation_path), decision=decision, checkpoint=candidate)
      )
      if decision["stop"]:
        stopped = True
        self.state.update(
          stop_reasons=decision["reasons"], rollback_checkpoint=accepted
        )
        self.save()
        break
      if not decision["eligible"]:
        raise RuntimeError("Gate returned neither eligible nor stop")
      accepted, accepted_path = candidate, validation_path
      self.state["accepted_checkpoint"] = accepted
      previous = current
      self.save()
    # Freeze selection before either evaluation on the final held-out protocol.
    self.state.update(
      selected_checkpoint=accepted,
      selected_validation=str(accepted_path),
      selection_finished_at=time.time(),
    )
    self.save()
    final_path = self.output / "final_holdout.json"
    baseline_final_path = self.output / "baseline_final_holdout.json"
    selected_final = self.evaluate(accepted, final_path, "final")
    baseline_final = self.evaluate(
      self.plan["baseline_checkpoint"], baseline_final_path, "final"
    )
    comparison_path = self.output / "paired_final_summary.json"
    atomic_json(comparison_path, paired_final_report(baseline_final, selected_final))
    self.state.update(
      state="stopped_safely" if stopped else "completed",
      final_holdout=str(final_path),
      baseline_final_holdout=str(baseline_final_path),
      paired_final_summary=str(comparison_path),
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
    source_record = checkpoint_record(SOURCE)
    if source_record["sha256"] != SOURCE_SHA256:
      raise RuntimeError("Continuation source hash does not match the approved source")
    frozen = {
      **plan,
      "sources": source_manifest(),
      "source_checkpoint": source_record,
      "baseline_checkpoint": checkpoint_record(BASELINE),
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
  parser.add_argument("--iterations", type=int, default=30000)
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
