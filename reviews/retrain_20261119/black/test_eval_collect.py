"""合成采集验证首次episode及失败分母；不创建物理环境。"""

import json
from types import SimpleNamespace

import pytest
import torch

import black.eval_collect as collector
from black.eval_collect import close_environment, collect_batch, new_trial
from black.eval_protocol import (
  PROTOCOL,
  PROTOCOL_ID,
  QUALIFICATION_ID,
  SCHEMA,
  command,
  padded_batch,
  roster,
)


def test_fixed_roster_and_command_boundaries():
  axes = roster("axes", 11)
  assert len(axes) == 24
  assert len({(tuple(row["target"]), row["orientation"]) for row in axes}) == 24
  assert all(row["seed"] == 11 and row["horizon"] == 900 for row in axes)
  spec = axes[0]
  assert command(spec, 99) == [0, 0, 0]
  assert command(spec, 100)[0] < 0
  assert command(spec, 350)[0] > 0
  assert command(spec, 600) == [0, 0, 0]
  stairs = roster("stairs", 11)
  assert len(stairs) == 54 and stairs[32]["seed"] == 12
  assert stairs[32]["lane"] == 0
  padded = padded_batch(stairs[32:])
  assert len(padded) == 32 and sum(row["padding"] for row in padded) == 10
  assert PROTOCOL["qualification"] == QUALIFICATION_ID


class FakeEnv:
  def __init__(self):
    self.tick = 0
    self.probe = None
    self.camera_receipt = {}
    self.render_receipt = {}
    self.closed = []
    self.frames = SimpleNamespace(episode=torch.zeros(2, dtype=torch.long))
    self.frames.metadata = lambda: dict(
      frame=torch.zeros(2, dtype=torch.long),
      capture_tick=torch.zeros(2, dtype=torch.long),
      mask=torch.ones(2, dtype=torch.bool),
    )
    self.evaluation = self
    self.obs = dict(history=torch.zeros(2, 225), geometry=torch.zeros(2, 768))
    self.record = None

  def pre(self) -> dict:
    return dict(
      tick=self.tick,
      episode=self.frames.episode.clone(),
      command=torch.zeros(2, 3),
      velocity=torch.zeros(2, 3),
      tilt=torch.zeros(2),
      foot_contacts=torch.ones(2, 4, dtype=torch.bool),
      support=torch.ones(2, dtype=torch.bool),
    )

  def step(self, action):
    terminal = torch.tensor([self.tick == 1, False])
    self.record = dict(
      terminated=terminal,
      truncated=torch.zeros(2, dtype=torch.bool),
      failed=terminal.clone(),
      action=action.clone(),
      raw_action=action.clone(),
      delayed_action=action.clone(),
      processed_target=action.clone(),
      action_delay=torch.zeros(2, dtype=torch.long),
    )
    self.tick += 1
    self.frames.episode += terminal.long()

  def post(self):
    return self.record

  def close(self):
    self.closed.append(True)


def trials_for(specs):
  manifest = dict(
    schema=SCHEMA,
    protocol_id=PROTOCOL_ID,
    teacher_id="teacher",
    bundle_sha256="bundle",
    source_sha256="source",
  )
  return [new_trial(spec, manifest) for spec in specs]


def test_first_terminal_is_retained_without_reset_rows_or_padding():
  specs = roster("flat8", 11)[:2]
  trials = trials_for(specs)
  env = FakeEnv()
  collect_batch(env, lambda *args: torch.zeros(2, 12), specs, trials, max_controls=4)
  assert len(trials) == 2
  assert len(trials[0]["rows"]) == 2 and len(trials[1]["rows"]) == 4
  assert trials[0]["end_reason"] == "physical_failure"
  assert trials[1]["end_reason"] == "administrative_incomplete"
  assert trials[0]["rows"][-1]["post_episode"] == 1
  assert all(row["episode"] == 0 for row in trials[0]["rows"])


def test_nonfinite_action_preserves_unstarted_physics_and_partial_trial():
  specs = roster("flat8", 11)[:2]
  trials = trials_for(specs)
  env = FakeEnv()
  with pytest.raises(FloatingPointError, match="非有限"):
    collect_batch(env, lambda *args: torch.full((2, 12), float("nan")), specs, trials)
  assert env.tick == 0
  assert all(not trial["rows"] for trial in trials)


def test_cleanup_failure_keeps_original_error():
  original = ValueError("first")

  def close():
    raise RuntimeError("close")

  env = SimpleNamespace(close=close)
  close_environment(env, original)
  assert original.__dict__["failure_details"] == {
    "close_error": "RuntimeError('close')"
  }
  with pytest.raises(RuntimeError, match="close"):
    close_environment(env)


def test_nonfinite_measurement_stops_before_physics(monkeypatch):
  env = FakeEnv()
  original = env.pre

  def pre():
    value = original()
    value["velocity"][0, 0] = float("nan")
    return value

  monkeypatch.setattr(env, "pre", pre)
  specs = roster("flat8", 11)[:2]
  with pytest.raises(FloatingPointError, match="动作前测量"):
    collect_batch(env, lambda *args: torch.zeros(2, 12), specs, trials_for(specs))
  assert env.tick == 0


def test_json_publication_never_leaves_half_result(tmp_path):
  path = tmp_path / "RESULT.json"
  with pytest.raises(ValueError):
    collector.write_json(path, {"value": float("nan")})
  assert not list(tmp_path.iterdir())
  collector.write_json(path, {"complete": True})
  before = path.read_bytes()
  with pytest.raises(FileExistsError):
    collector.write_json(path, {"complete": False})
  assert path.read_bytes() == before
  assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("failed_path", ["trial_0001.json", "RESULT.json", "partial"])
def test_publication_failure_retains_trials_and_failed_record(
  tmp_path, monkeypatch, failed_path
):
  class Teacher:
    def freeze(self):
      pass

    def __call__(self, *args):
      return torch.zeros(2, 12)

  env = FakeEnv()
  closed = env.closed
  monkeypatch.setattr(
    collector,
    "load_bundle",
    lambda *args: (Teacher(), None, None, {"completed_updates": 2}),
  )
  monkeypatch.setattr(collector, "teacher_identity", lambda *args: "teacher")
  monkeypatch.setattr(collector, "source_files", lambda: {})
  monkeypatch.setattr(collector, "NativeEnv", lambda *args, **kwargs: env)
  original = collector.write_json
  error = OSError("publication failed")

  def write(path, value):
    if path.name == failed_path or (
      failed_path == "partial" and path.name in ("RESULT.json", "PARTIAL.json")
    ):
      raise error
    return original(path, value)

  monkeypatch.setattr(collector, "write_json", write)
  bundle = tmp_path / "bundle.pt"
  bundle.write_bytes(b"synthetic bundle")
  output = tmp_path / "run"
  args = SimpleNamespace(
    out=output, bundle=bundle, device="cpu", suite="flat8", seed=11, max_controls=4
  )
  specs = roster("flat8", 11)[:2]
  with pytest.raises(OSError) as raised:
    collector.execute(args, specs)
  assert raised.value is error
  assert closed == [True]
  assert not (output / "RESULT.json").exists()
  failure = json.loads((output / "FAILED.json").read_text())
  assert failure["error"] == repr(error)
  if failed_path == "partial":
    assert failure["partial_write_error"] == repr(error)
  else:
    partial = json.loads((output / "PARTIAL.json").read_text())
    assert len(partial["roster"]) == len(partial["trials"]) == 2
    assert [len(trial["rows"]) for trial in partial["trials"]] == [2, 4]
