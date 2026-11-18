"""CPU-only serial lineage and safety checks; no simulator runs."""

from contextlib import nullcontext
from pathlib import Path

import pytest

from mjlab.scripts import run_black_continuation as serial


def test_budget_and_dry_plan(tmp_path):
  plan = serial.make_plan(tmp_path / "absent", 10000, 1000)
  assert len(plan["chunks"]) == 10
  assert plan["chunks"][-1] == dict(offset=9000, updates=1000)
  assert not (tmp_path / "absent").exists()
  assert serial.chunk_schedule(1001, 1000) == [(0, 1000), (1000, 1)]
  for budget, chunk in [(0, 1000), (10001, 1000), (10000, 1001)]:
    with pytest.raises(ValueError):
      serial.chunk_schedule(budget, chunk)


def test_existing_output_refused_before_guard(tmp_path, monkeypatch):
  monkeypatch.setattr(serial, "lock", lambda *_: nullcontext(-1))
  monkeypatch.setattr(serial, "guard_resources", lambda *_: pytest.fail("guard called"))
  with pytest.raises(FileExistsError):
    serial.execute_plan(serial.make_plan(tmp_path, 1, 1))


@pytest.mark.parametrize("stop_stage", [None, 1, 2])
def test_serial_selection_and_safe_stop(tmp_path, monkeypatch, stop_stage):
  from mjlab.scripts import evaluate_black_continuation as evaluator

  source = tmp_path / "source.pt"
  source.write_bytes(b"source")
  output = tmp_path / "run"
  output.mkdir()
  plan = serial.make_plan(output, 2, 1)
  plan.update(source_checkpoint=serial.checkpoint_record(source), fingerprint="test")
  supervisor = serial.Supervisor(plan, -1)
  commands, evaluations, gates = [], [], []

  def run_child(command, label, seed):
    commands.append(command)
    directory = Path(command[command.index("--output") + 1])
    directory.mkdir(parents=True)
    checkpoint = directory / "model.pt"
    checkpoint.write_bytes(str(directory).encode())
    offset = int(command[command.index("--stage-offset") + 1])
    serial.atomic_json(
      directory / "status.json",
      dict(
        state="completed",
        completed_stage_updates=offset + 1,
        checkpoint=serial.checkpoint_record(checkpoint),
      ),
    )

  def evaluate(record, path, split="validation"):
    report = dict(checkpoint=record, path=str(path), split=split)
    evaluations.append(report)
    serial.atomic_json(path, report)
    return report

  def gate(reference, current, previous=None, stage1_reference=None):
    gates.append((current, previous, stage1_reference))
    stop = bool(stop_stage and f"stage{stop_stage}" in current["path"])
    return dict(stop=stop, eligible=not stop, reasons=["regression"] if stop else [])

  monkeypatch.setattr(supervisor, "run_child", run_child)
  monkeypatch.setattr(supervisor, "evaluate", evaluate)
  monkeypatch.setattr(evaluator, "gate_checkpoint", gate)
  supervisor.execute()
  assert evaluations[-1]["split"] == "final"
  assert all(g[0]["split"] != "final" for g in gates)
  assert supervisor.state["selected_checkpoint"] == evaluations[-1]["checkpoint"]
  stages = [int(c[c.index("--stage") + 1]) for c in commands]
  if stop_stage == 1:
    assert stages == [1]
    assert supervisor.state["selected_checkpoint"] == plan["source_checkpoint"]
  else:
    assert stages[:2] == [1, 1]
    stage2 = commands[2]
    assert "--parent-validation" in stage2
    assert (
      stage2[stage2.index("--source") + 1]
      == (supervisor.state["selected_stage1_checkpoint"]["path"])
    )
    assert gates[3][2] is not None
  if stop_stage:
    assert supervisor.state["state"] == "stopped_safely"
  else:
    assert stages == [1, 1, 2, 2]
    assert supervisor.state["state"] == "completed"
  assert source.read_bytes() == b"source"
