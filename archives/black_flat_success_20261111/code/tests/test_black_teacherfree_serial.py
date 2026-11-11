"""CPU-only fake subprocess tests for single-lineage teacher-free scheduling."""

from contextlib import nullcontext
from pathlib import Path

import pytest

from mjlab.scripts import run_black_teacherfree as serial


def test_budget_and_dry_plan(tmp_path):
  plan = serial.make_plan(tmp_path / "absent", 30000, 1000)
  assert len(plan["chunks"]) == 30
  assert sum(c["updates"] for c in plan["chunks"]) == 30000
  assert plan["chunks"][-1] == dict(offset=29000, updates=1000)
  assert "stages" not in plan and plan["num_envs"] == 4096
  assert plan["source"].endswith("stage1/updates_01000/model_3999.pt")
  assert plan["baseline"].endswith("model_2999.pt")
  assert not (tmp_path / "absent").exists()
  assert serial.chunk_schedule(1001, 1000) == [(0, 1000), (1000, 1)]
  for budget, chunk in [(0, 1000), (30001, 1000), (30000, 1001), (1, 0)]:
    with pytest.raises(ValueError):
      serial.chunk_schedule(budget, chunk)


def test_existing_output_refused_before_guard(tmp_path, monkeypatch):
  monkeypatch.setattr(serial, "lock", lambda *_: nullcontext(-1))
  monkeypatch.setattr(serial, "guard_resources", lambda *_: pytest.fail("guard called"))
  with pytest.raises(FileExistsError):
    serial.execute_plan(serial.make_plan(tmp_path, 1, 1))


def fixture_supervisor(tmp_path):
  source = tmp_path / "source.pt"
  baseline = tmp_path / "baseline.pt"
  source.write_bytes(b"source")
  baseline.write_bytes(b"baseline")
  output = tmp_path / "run"
  output.mkdir()
  plan = serial.make_plan(output, 3, 1)
  plan.update(
    source_checkpoint=serial.checkpoint_record(source),
    baseline_checkpoint=serial.checkpoint_record(baseline),
    fingerprint="test",
  )
  return serial.Supervisor(plan, -1)


@pytest.mark.parametrize("stop_chunk", [None, 1, 2])
@pytest.mark.parametrize("weak_source", [False, True])
def test_serial_selection_and_safe_stop(tmp_path, monkeypatch, stop_chunk, weak_source):
  from mjlab.scripts import evaluate_black_teacherfree as evaluator

  supervisor = fixture_supervisor(tmp_path)
  commands, evaluations, gates = [], [], []

  def run_child(command, label, seed):
    commands.append(command)
    directory = Path(command[command.index("--output") + 1])
    directory.mkdir(parents=True)
    checkpoint = directory / "model.pt"
    checkpoint.write_bytes(str(directory).encode())
    offset = int(command[command.index("--offset") + 1])
    assert seed == 51001 + offset
    serial.atomic_json(
      directory / "status.json",
      dict(
        state="completed",
        completed_updates=offset + 1,
        global_iteration=4000 + offset,
        checkpoint={
          **serial.checkpoint_record(checkpoint),
          "global_iteration": 4000 + offset,
          "completed_updates": offset + 1,
        },
      ),
    )

  def evaluate(record, path, split="validation"):
    report = dict(
      checkpoint=serial.checkpoint_record(Path(record["path"])),
      path=str(path),
      split=split,
    )
    if split == "final":
      assert supervisor.state["selection_finished_at"]
      assert supervisor.state["selected_checkpoint"]
    evaluations.append(report)
    serial.atomic_json(path, report)
    return report

  def gate(reference, current, previous=None, stage1_reference=None):
    gates.append((current, previous, stage1_reference))
    stop = bool(
      (stop_chunk and f"updates_{stop_chunk:05d}" in current["path"])
      or (weak_source and "source_validation" in current["path"])
    )
    return dict(stop=stop, eligible=not stop, reasons=["regression"] if stop else [])

  monkeypatch.setattr(supervisor, "run_child", run_child)
  monkeypatch.setattr(supervisor, "evaluate", evaluate)
  monkeypatch.setattr(evaluator, "gate_checkpoint", gate)
  monkeypatch.setattr(evaluator, "_verified", lambda *_: True)
  monkeypatch.setattr(
    evaluator, "paired_final_report", lambda ref, cur: dict(reference=ref, selected=cur)
  )
  supervisor.execute()
  count = stop_chunk or 3
  assert len(commands) == count
  assert supervisor.state["completed_updates"] == count
  assert supervisor.state["global_iteration"] == 3999 + count
  assert all("--stage" not in c and "--parent-validation" not in c for c in commands)
  assert all(c[2] == "mjlab.scripts.train_black_teacherfree" for c in commands)
  assert (
    commands[0][commands[0].index("--source") + 1]
    == supervisor.plan["source_checkpoint"]["path"]
  )
  for i, command in enumerate(commands[1:], 1):
    assert (
      command[command.index("--source") + 1] == gates[i + 1][0]["checkpoint"]["path"]
    )
  assert gates[2][1] is None
  assert all(g[2] == evaluations[1] for g in gates[2:])
  if count > 1:
    assert gates[3][1] == gates[2][0]
  assert all(g[0]["split"] == "validation" for g in gates)
  assert [e["split"] for e in evaluations[-2:]] == ["final", "final"]
  assert evaluations[-1]["checkpoint"] == supervisor.plan["baseline_checkpoint"]
  assert (
    serial.checkpoint_record(Path(supervisor.state["selected_checkpoint"]["path"]))
    == evaluations[-2]["checkpoint"]
  )
  if stop_chunk == 1:
    assert (
      supervisor.state["selected_checkpoint"] == supervisor.plan["source_checkpoint"]
    )
  elif stop_chunk == 2:
    assert (
      serial.checkpoint_record(Path(supervisor.state["selected_checkpoint"]["path"]))
      == gates[2][0]["checkpoint"]
    )
  else:
    assert (
      serial.checkpoint_record(Path(supervisor.state["selected_checkpoint"]["path"]))
      == gates[-1][0]["checkpoint"]
    )
  assert supervisor.state["state"] == ("stopped_safely" if stop_chunk else "completed")
  assert Path(supervisor.plan["source_checkpoint"]["path"]).read_bytes() == b"source"


@pytest.mark.parametrize(
  "bad_field", ["state", "completed_updates", "global_iteration"]
)
def test_incomplete_child_is_rejected(tmp_path, monkeypatch, bad_field):
  from mjlab.scripts import evaluate_black_teacherfree as evaluator

  supervisor = fixture_supervisor(tmp_path)
  monkeypatch.setattr(supervisor, "evaluate", lambda *a, **kw: {})
  monkeypatch.setattr(evaluator, "_verified", lambda *_: True)
  monkeypatch.setattr(evaluator, "gate_checkpoint", lambda *a, **kw: dict(stop=False))

  def child(command, *_):
    directory = Path(command[command.index("--output") + 1])
    directory.mkdir()
    status = dict(state="completed", completed_updates=1, global_iteration=4000)
    status[bad_field] = "bad"
    serial.atomic_json(directory / "status.json", status)

  monkeypatch.setattr(supervisor, "run_child", child)
  with pytest.raises(RuntimeError, match="cumulative updates"):
    supervisor.execute()
  assert "selected_checkpoint" not in supervisor.state


def test_source_hash_mismatch_refused(tmp_path, monkeypatch):
  source = tmp_path / "source.pt"
  source.write_bytes(b"wrong checkpoint")
  output = tmp_path / "run"
  monkeypatch.setattr(serial, "SOURCE", source)
  monkeypatch.setattr(serial, "lock", lambda *_: nullcontext(-1))
  monkeypatch.setattr(serial, "guard_resources", lambda *_: None)
  with pytest.raises(RuntimeError, match="source hash"):
    serial.execute_plan(serial.make_plan(output, 1, 1))
  assert not output.exists()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_subprocess_inherits_lock_and_clears_pid(tmp_path, monkeypatch, exit_code):
  from mjlab.scripts import run_black_continuation as frozen

  supervisor = fixture_supervisor(tmp_path)
  supervisor.fd = 123
  supervisor.plan["sources"] = {}
  calls = []
  guards = []

  class FakeProcess:
    pid = 987654
    returncode = exit_code

    def wait(self, timeout=None):
      return exit_code

    def poll(self):
      return exit_code

  def popen(command, **kwargs):
    calls.append((command, kwargs))
    return FakeProcess()

  monkeypatch.setenv("RANK", "1")
  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setattr(frozen, "verify_sources", lambda *_: None)
  monkeypatch.setattr(frozen, "guard_resources", lambda *args: guards.append(args))
  monkeypatch.setattr(frozen.subprocess, "Popen", popen)
  if exit_code:
    with pytest.raises(RuntimeError, match="exited 7"):
      supervisor.run_child(["fake-child"], "fake", 51002)
  else:
    supervisor.run_child(["fake-child"], "fake", 51002)
  assert len(calls) == len(guards) == 1
  kwargs = calls[0][1]
  assert kwargs["pass_fds"] == (123,)
  assert kwargs["start_new_session"]
  assert "RANK" not in kwargs["env"] and "WORLD_SIZE" not in kwargs["env"]
  assert kwargs["env"]["PYTHONHASHSEED"] == "51002"
  assert supervisor.state["child_pid"] is None
  assert supervisor.state["child_exit_code"] == exit_code


def test_spawn_failure_is_not_retried(tmp_path, monkeypatch):
  from mjlab.scripts import run_black_continuation as frozen

  supervisor = fixture_supervisor(tmp_path)
  supervisor.plan["sources"] = {}
  calls = []

  def popen(*args, **kwargs):
    calls.append(args)
    raise OSError("spawn failed")

  monkeypatch.setattr(frozen, "verify_sources", lambda *_: None)
  monkeypatch.setattr(frozen, "guard_resources", lambda *_: None)
  monkeypatch.setattr(frozen.subprocess, "Popen", popen)
  with pytest.raises(OSError, match="spawn failed"):
    supervisor.run_child(["fake-child"], "fake", 51001)
  assert len(calls) == 1
  assert supervisor.state["child_pid"] is None


@pytest.mark.parametrize("valid,matching", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("with_metadata", [False, True])
def test_evaluator_protocol_and_checkpoint_checks(
  tmp_path, monkeypatch, valid, matching, with_metadata
):
  from mjlab.scripts import evaluate_black_teacherfree as evaluator

  supervisor = fixture_supervisor(tmp_path)
  record = dict(supervisor.plan["source_checkpoint"])
  expected = dict(record)
  if with_metadata:
    record.update(global_iteration=3999, completed_updates=0)
  output = supervisor.output / "validation.json"
  seen = []

  def child(command, label, seed):
    seen.append((command, seed))
    serial.atomic_json(output, dict(checkpoint=expected if matching else {}))

  monkeypatch.setattr(supervisor, "run_child", child)
  monkeypatch.setattr(evaluator, "_verified", lambda *a, **kw: valid)
  if valid and matching:
    assert supervisor.evaluate(record, output)["checkpoint"] == expected
  else:
    with pytest.raises(RuntimeError, match="incomplete or mismatched"):
      supervisor.evaluate(record, output)
  assert seen[0][0][2] == "mjlab.scripts.evaluate_black_teacherfree"
  assert seen[0][1] == 941101
