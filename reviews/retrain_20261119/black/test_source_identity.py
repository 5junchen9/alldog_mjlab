"""资产/运行库改变必须换身份；目录搬迁不能改变同一份证据身份。"""

import shutil

import pytest

import black.eval_collect as collector
from black.eval_protocol import digest


def select_root(monkeypatch, root):
  base = root / "support" / "black"
  monkeypatch.setattr(collector, "ROOT", root)
  monkeypatch.setattr(collector, "BASE", base)
  monkeypatch.setattr(collector, "HERE", root / "black")


@pytest.mark.parametrize("suffix", [".xml", ".urdf", ".STL", ".obj", ".png"])
def test_asset_content_change_changes_identity(tmp_path, monkeypatch, suffix):
  select_root(monkeypatch, tmp_path)
  asset = tmp_path / "src" / "robot" / f"asset{suffix}"
  asset.parent.mkdir(parents=True)
  asset.write_bytes(b"original asset")
  before = collector.source_files()
  key = asset.relative_to(tmp_path).as_posix()
  assert before[key] == collector.sha256(asset)
  asset.write_bytes(b"changed collision, inertia, or image")
  after = collector.source_files()
  assert before[key] != after[key]
  assert digest(before) != digest(after)


@pytest.mark.parametrize(
  "package",
  ["torch", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "numpy", "tensordict"],
)
def test_runtime_package_version_change_changes_identity(
  tmp_path, monkeypatch, package
):
  select_root(monkeypatch, tmp_path)
  versions = dict(collector.runtime_identity()["packages"])
  monkeypatch.setattr(collector, "version", lambda name: versions[name])
  before = collector.source_files()
  versions[package] += ".changed"
  after = collector.source_files()
  assert before["@runtime/versions.json"] != after["@runtime/versions.json"]
  assert digest(before) != digest(after)


def test_python_and_cuda_build_versions_are_bound(tmp_path, monkeypatch):
  select_root(monkeypatch, tmp_path)
  before = collector.source_files()
  monkeypatch.setattr(collector, "python_version", lambda: "different Python")
  after_python = collector.source_files()
  assert digest(before) != digest(after_python)
  monkeypatch.setattr(collector.torch.version, "cuda", "different CUDA build")
  assert digest(after_python) != digest(collector.source_files())


def test_relocated_roots_keep_identity_and_dynamic_dependency_is_bound(
  tmp_path, monkeypatch
):
  first, second = tmp_path / "first", tmp_path / "relocated"
  relative_files = (
    "src/robot/model.py",
    "src/robot/black.urdf",
    "src/robot/meshes/body.STL",
    "black/runtime.py",
    "support/black/config.py",
    "support/black/specialty_composition.py",
    "support/black/obstacle_geometry.py",
  )
  for relative in relative_files:
    path = first / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(relative)
  shutil.copytree(first, second)
  select_root(monkeypatch, first)
  original = collector.source_files()
  assert set(original) == {*relative_files, "@runtime/versions.json"}
  select_root(monkeypatch, second)
  relocated = collector.source_files()
  assert original == relocated
  assert digest(original) == digest(relocated)
  (second / relative_files[-1]).write_text("changed dynamic dependency")
  assert digest(original) != digest(collector.source_files())
