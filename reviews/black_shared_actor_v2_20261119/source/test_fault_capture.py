"""首坏诊断的完整发布和失败保留。"""

from types import SimpleNamespace

import pytest
import torch
from fault_capture import write_fault


def test_fault_publishes_complete_file_without_overwrite(tmp_path):
  probe = SimpleNamespace(fault={"qpos": torch.arange(3.0), "replayable": False})
  target = write_fault(probe, tmp_path / "fault")
  packet = torch.load(target, weights_only=True)
  torch.testing.assert_close(packet["qpos"], probe.fault["qpos"])
  assert not packet["replayable"]
  with pytest.raises(FileExistsError):
    write_fault(probe, tmp_path / "fault")


def test_fault_write_error_has_no_published_half_packet(tmp_path, monkeypatch):
  def broken_save(value, stream):
    stream.write(b"incomplete")
    raise OSError("disk full")

  monkeypatch.setattr(torch, "save", broken_save)
  with pytest.raises(OSError, match="disk full"):
    write_fault(SimpleNamespace(fault={"qpos": torch.ones(3)}), tmp_path / "fault")
  assert list((tmp_path / "fault").iterdir()) == []
