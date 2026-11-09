"""Explicit target-space adapters, never an assertion of dynamic equivalence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ControlContract:
  joint_names: tuple[str, ...]
  kinds: tuple[str, ...]
  scales: tuple[float, ...]
  offsets: tuple[float, ...]
  stiffness: tuple[float, ...]
  damping: tuple[float, ...]
  control_dt: float

  def __post_init__(self):
    n = len(self.joint_names)
    if n != 16 or len(set(self.joint_names)) != n:
      raise ValueError("Wolf requires 16 unique joint names")
    for name in ("kinds", "scales", "offsets", "stiffness", "damping"):
      if len(getattr(self, name)) != n:
        raise ValueError(f"{name} must have {n} entries")
    if not set(self.kinds) <= {"position", "velocity"}:
      raise ValueError("unsupported control kind")
    for values in (self.scales, self.offsets, self.stiffness, self.damping):
      if not all(math.isfinite(v) for v in values):
        raise ValueError("control contract must be finite")
    if any(v == 0 for v in self.scales):
      raise ValueError("action scales must be nonzero")
    if not math.isfinite(self.control_dt) or self.control_dt <= 0:
      raise ValueError("control_dt must be positive finite")

  @classmethod
  def from_dict(cls, value: dict[str, Any]) -> ControlContract:
    return cls(
      joint_names=tuple(value["joint_names"]),
      kinds=tuple(value["kinds"]),
      scales=tuple(value["scales"]),
      offsets=tuple(value["offsets"]),
      stiffness=tuple(value["stiffness"]),
      damping=tuple(value["damping"]),
      control_dt=float(value["control_dt"]),
    )

  def digest(self) -> str:
    return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class TargetSpaceAdapter(nn.Module):
  """Raw source action -> nominal physical target -> destination raw action.

  This does not translate PD forces, delays, DR, clipping, friction or dynamics.
  Different dt is rejected; gains are preserved as evidence, not compensated.
  No clipping hides an out-of-domain mapping.
  """

  indices: torch.Tensor
  source_scale: torch.Tensor
  source_offset: torch.Tensor
  target_scale: torch.Tensor
  target_offset: torch.Tensor

  def __init__(self, source: ControlContract, target: ControlContract):
    super().__init__()
    if set(source.joint_names) != set(target.joint_names):
      raise ValueError("joint names do not match")
    if source.control_dt != target.control_dt:
      raise ValueError("different control_dt requires a separate verified adapter")
    indices = [source.joint_names.index(name) for name in target.joint_names]
    if any(
      source.kinds[i] != kind for i, kind in zip(indices, target.kinds, strict=True)
    ):
      raise ValueError("position/velocity contracts cannot be interchanged")
    self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
    self.register_buffer("source_scale", torch.tensor(source.scales)[indices])
    self.register_buffer("source_offset", torch.tensor(source.offsets)[indices])
    self.register_buffer("target_scale", torch.tensor(target.scales))
    self.register_buffer("target_offset", torch.tensor(target.offsets))

  def forward(self, actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[-1] != 16 or not torch.isfinite(actions).all():
      raise ValueError("actions must be finite with 16 channels")
    return (
      actions.index_select(-1, self.indices) * self.source_scale
      + self.source_offset
      - self.target_offset
    ) / self.target_scale


def sha256_file(path: Path) -> str:
  with path.open("rb") as stream:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
      digest.update(block)
    return digest.hexdigest()


def adapter_digest(manifest: dict[str, Any]) -> str:
  """Bind reports to the exact observation/action/history bridge, not just shapes."""
  keys = (
    "source_control",
    "observation_matrix",
    "observation_bias",
    "input_group",
    "history_length",
    "history_order",
    "observation_contract",
    "normalization_contract",
    "history_contract",
  )
  payload = {key: manifest[key] for key in keys}
  return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verify_teacher_manifest(
  manifest: dict[str, Any], checkpoint: Path, target: ControlContract, task: str
) -> None:
  """Fail closed unless recorded contract and closed-loop evidence match bytes.

  The caller supplies an audited manifest. This checks consistency, not the
  truthfulness of external reports, and does not produce a qualification.
  """
  if manifest.get("schema_version") != 1:
    raise ValueError("teacher manifest schema_version must be 1")
  if manifest.get("teacher_sha256") != sha256_file(checkpoint):
    raise ValueError("teacher checkpoint hash mismatch")
  if manifest.get("qualification") != "qualified":
    raise ValueError("teacher is unqualified; fixtures are not teachers")
  if manifest.get("target_contract_sha256") != target.digest():
    raise ValueError("target control contract mismatch")
  if manifest.get("target_task") != task:
    raise ValueError("teacher qualification does not cover target task")
  source = ControlContract.from_dict(manifest["source_control"])
  TargetSpaceAdapter(source, target)
  for key in ("observation_contract", "normalization_contract", "history_contract"):
    if not isinstance(manifest.get(key), str) or not manifest[key].strip():
      raise ValueError(f"missing audited {key}")
  if manifest.get("curriculum_policy") != "frozen_initial_config":
    raise ValueError("teacher qualification must cover the fixed distillation domain")
  evidence = manifest.get("evidence", {})
  required = {"same_state_observation", "closed_loop_target"}
  if source.digest() != target.digest():
    required.add("cross_contract_mapping")
  for key in required:
    item = evidence.get(key, {})
    path = Path(item.get("path", ""))
    if not path.is_file() or item.get("sha256") != sha256_file(path):
      raise ValueError(f"missing or changed qualification evidence: {key}")
    report = json.loads(path.read_text())
    expected = {
      "passed": True,
      "validation_kind": key,
      "teacher_sha256": manifest["teacher_sha256"],
      "target_contract_sha256": target.digest(),
      "target_task": task,
      "target_runtime_sha256": manifest["target_runtime_sha256"],
      "adapter_sha256": adapter_digest(manifest),
      "curriculum_policy": "frozen_initial_config",
    }
    if any(report.get(field) != value for field, value in expected.items()):
      raise ValueError(f"qualification evidence does not cover this adapter: {key}")
  if manifest.get("format") != "torchscript_tensor":
    raise ValueError("export a stateless tensor-input TorchScript teacher first")
  if manifest.get("input_group") not in ("actor", "critic"):
    raise ValueError("teacher input_group must be actor or critic")
  if manifest.get("history_order") not in ("newest_first", "oldest_first"):
    raise ValueError("explicit history_order is required")
  if manifest.get("input_group") == "critic" and manifest.get("history_length") != 1:
    raise ValueError("critic history is not implemented")


class FrozenTeacher(nn.Module):
  """Tensor-only teacher with audited affine observation adaptation.

  Input frames already refer to the current simulator state. An affine map may
  reorder, rescale, offset, and convert last-action semantics. Any nonlinear,
  delayed or recurrent observation bridge needs a separate implementation.
  The frozen normalizer must be embedded in the teacher export.
  """

  matrix: torch.Tensor
  bias: torch.Tensor

  def __init__(
    self,
    model: nn.Module,
    adapter: TargetSpaceAdapter,
    matrix: torch.Tensor,
    bias: torch.Tensor,
    history_length: int = 1,
    newest_first: bool = True,
  ):
    super().__init__()
    if matrix.ndim != 2 or bias.shape != matrix.shape[:1]:
      raise ValueError("observation matrix/bias shape mismatch")
    if not torch.isfinite(matrix).all() or not torch.isfinite(bias).all():
      raise ValueError("observation adapter must be finite")
    if history_length < 1:
      raise ValueError("history_length must be positive")
    self.model = model
    self.adapter = adapter
    self.register_buffer("matrix", matrix)
    self.register_buffer("bias", bias)
    self.history_length = history_length
    self.newest_first = newest_first
    self.requires_grad_(False)
    self.eval()

  def train(self, mode: bool = True):
    # Parent .train() must not update the teacher normalizer/dropout.
    return super().train(False)

  @torch.no_grad()
  def forward(self, frames: torch.Tensor) -> torch.Tensor:
    frames = frames.unsqueeze(1) if frames.ndim == 2 else frames
    if frames.ndim != 3 or frames.shape[1] != self.history_length:
      raise ValueError("teacher history length mismatch")
    value = frames @ self.matrix.T + self.bias
    if self.newest_first:
      value = value.flip(1)
    return self.adapter(self.model(value.flatten(1)))
