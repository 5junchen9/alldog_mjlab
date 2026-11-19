from copy import deepcopy

import numpy as np
import pytest
import torch
from black.checkpoint import tensor_digest
from black.run import update_him
from black.student import SharedStudent, distillation_loss
from tensordict import TensorDict
from wolf.checkpoint import CONTRACT, load_bundle, save_bundle
from wolf.curriculum import COURSES, SPECIALTIES, configure, flat_command
from wolf.eval_protocol import SEEDS, panel, requirements
from wolf.eval_score import geometry_contract, score_trial
from wolf.model import initial_models
from wolf.ppo import build_ppo


def test_shared_actor_gradient_paths():
  torch.set_num_threads(2)
  teacher, cnn, gru = initial_models(19)
  history, geometry = torch.randn(3, 6, 53), torch.randn(3, 768)
  student = SharedStudent(teacher, cnn, gru)
  visual = torch.randn(3, 32, requires_grad=True)
  expected = teacher(history, geometry)
  torch.testing.assert_close(
    expected,
    student.act_with_visual(history, teacher.terrain_encoder(geometry)),
    atol=0,
    rtol=0,
  )
  loss, _ = distillation_loss(history, geometry, visual, student, teacher, 1.0)
  loss.backward()
  assert visual.grad.abs().sum() > 0
  assert all(p.grad is None for p in teacher.parameters())
  assert student.actor is teacher.actor and student.him is teacher.him


def test_him_update_only_changes_him():
  teacher, _, _ = initial_models(20)
  before_actor = tensor_digest(teacher.actor.state_dict())
  before_him = tensor_digest(teacher.him.state_dict())
  optimizer = torch.optim.Adam(teacher.him.parameters(), lr=1e-4)
  sample = [(torch.randn(8, 6, 53), torch.randn(8, 53), torch.randn(8, 3))]
  result = update_him(teacher.him, optimizer, sample, "cpu")
  assert result["optimizer_calls"] == 1
  assert tensor_digest(teacher.actor.state_dict()) == before_actor
  assert tensor_digest(teacher.him.state_dict()) != before_him
  assert all(not p.requires_grad for p in teacher.him.parameters())


def test_ppo_has_16_actions_and_no_him_parameters():
  teacher, _, _ = initial_models(21)
  obs = TensorDict(
    {
      "history": torch.zeros(4, 318),
      "geometry": torch.zeros(4, 768),
      "critic": torch.zeros(4, 430),
    },
    batch_size=[4],
  )
  ppo = build_ppo(obs, teacher)
  assert ppo.act(obs).shape == (4, 16)
  owned = {id(p) for group in ppo.optimizer.param_groups for p in group["params"]}
  assert not owned.intersection(id(p) for p in teacher.him.parameters())


def test_checkpoint_rejects_old_and_other_robot(tmp_path, monkeypatch):
  monkeypatch.setattr(
    "wolf.checkpoint.source_identity",
    lambda: {"sha256": "same_code_assets", "runtime": {"torch": "test_original"}},
  )
  teacher, cnn, gru = initial_models(22)
  path = tmp_path / "new.pt"
  save_bundle(path, teacher, cnn, gru, phase="probe", updates=0)
  restored, _, _, packet = load_bundle(path)
  assert tensor_digest(restored.state_dict()) == tensor_digest(teacher.state_dict())
  packet["contract"] = {"schema": "old_wolf"}
  torch.save(packet, tmp_path / "old.pt")
  with pytest.raises(ValueError, match="合同"):
    load_bundle(tmp_path / "old.pt")
  with pytest.raises(FileExistsError):
    save_bundle(path, teacher, cnn, gru, phase="probe", updates=0)


@pytest.mark.parametrize("course", COURSES)
def test_course_config_contract(course):
  cfg = configure(course, 32, 19)
  assert len(cfg.actions) == 8
  assert cfg.observations["actor"].history_length == 6
  assert "privileged_terrain" in cfg.observations["critic"].terms
  assert "privileged_terrain" not in cfg.observations["actor"].terms
  assert cfg.sim.mujoco.cone == "elliptic"
  assert cfg.curriculum == {}
  assert cfg.auto_reset is False
  assert "stand_still" not in cfg.rewards
  assert cfg.rewards["stop_wheels"].weight < 0
  if course == "general":
    assert list(cfg.scene.terrain.terrain_generator.sub_terrains) == [
      "flat",
      "slope_up",
      "slope_down",
      "stairs_up",
      "stairs_down",
      "rough",
      "wave",
    ]
    assert cfg.episode_length_s == 27
  if course in SPECIALTIES:
    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert cfg.scene.terrain.terrain_generator.sub_terrains["flat"].proportion == 0.25


def test_student_cap_and_complete_seed_panels():
  cfg = configure("stairs", max_level=2, allow_promotion=False)
  assert cfg.scene.terrain.max_init_terrain_level == 2
  assert cfg.commands["twist"].max_level == 2
  assert not cfg.commands["twist"].allow_promotion
  assert requirements("student:stairs") == (
    "flat_foundation",
    "flat_axes",
    "general",
    "stairs",
  )
  rows = [row for seed in SEEDS for row in panel("stairs", seed, 2)]
  assert len(rows) == 18
  assert {row["level"] for row in rows} == {0, 1, 2}
  assert len({row["trial_id"] for row in rows}) == len(rows)
  assert {row["domain"] for row in panel("general", SEEDS[0])} == set(range(1, 7))
  assert {row["level"] for row in panel("general", SEEDS[0])} == {1, 2, 3, 4}


def flat_rows(spec):
  rows = []
  target = torch.tensor([spec["target"]])
  for tick in range(spec["duration"]):
    command = flat_command(torch.tensor([tick]), target, spec["course"] == "general")[
      0
    ].tolist()
    rows.append(
      {
        "tick": tick,
        "control_tick": tick,
        "episode": 0,
        "capture_episode": 0,
        "frame": tick // 5,
        "capture_tick": tick // 5 * 5,
        "depth_valid": True,
        "depth_has_hit": True,
        "velocity": command[:2] + [0.0],
        "angular": [0.0, 0.0, command[2]],
        "tilt": 0.0,
        "command": command,
        "history_command": (np.asarray(command) * [2, 2, 0.25]).tolist(),
        "contacts": [True] * 4,
        "contact_points": [[0.0, 0.0, 0.0]] * 4,
        "root": [0.0, 0.0, 0.4],
        "action": [0.0] * 16,
        "received_action": [0.0] * 16,
        "raw_action": [0.0] * 16,
        "processed_targets": [0.0] * 16,
        "native_failure": False,
        "lower_floor": False,
        "timeout": False,
        "end": tick + 1 == spec["duration"],
      }
    )
  return rows


def test_raw_scorer_rejects_incomplete_spoof_and_failed_reverse():
  spec = panel("flat_foundation", SEEDS[0])[0]
  rows = flat_rows(spec)
  assert score_trial(rows, spec, None)["status"] == "pass"
  assert score_trial(rows[:100], spec, None)["status"] == "incomplete"
  altered = deepcopy(rows)
  altered[100]["received_action"][0] = 1.0
  assert score_trial(altered, spec, None)["status"] == "invalid"
  altered = deepcopy(rows)
  for row in altered[325:450]:
    row["velocity"] = [0.0, 0.0, 0.0]
  assert score_trial(altered, spec, None)["status"] == "fail"
  altered = deepcopy(rows)
  altered[-1]["tilt"] = 0.5
  assert score_trial(altered, spec, None)["status"] == "fail"


@pytest.mark.parametrize("course", SPECIALTIES)
def test_geometry_contract_preserves_real_cases(course):
  spec = panel(course, SEEDS[0], 0)[0]
  geometry = geometry_contract(spec)
  assert 0 < geometry["length"] < 30
  assert geometry["width"] > 0.1
  if course == "stairs":
    assert len(geometry["treads"]) > 2
  if course in ("broken_bridge", "plum_stakes"):
    assert geometry["surfaces"]


def test_qualification_recomputes_raw_and_enforces_complete_panel(
  tmp_path, monkeypatch
):
  import hashlib
  import json

  from wolf.eval_protocol import PROTOCOL
  from wolf.eval_score import digest, require_qualification

  identity = {"test_only": "cpu_fixture"}
  monkeypatch.setattr("wolf.eval_score.source_identity", lambda: identity)
  attempts = tmp_path / "attempt"
  attempts.mkdir()
  trials, hashes = [], {}
  for seed in SEEDS:
    for spec in panel("flat_foundation", seed):
      name = f"{len(trials):03d}.jsonl"
      path = attempts / name
      path.write_text("".join(json.dumps(row) + "\n" for row in flat_rows(spec)))
      trials.append({**spec, "raw_file": name})
      hashes[spec["trial_id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
  manifest = {
    "contract": CONTRACT,
    "protocol": PROTOCOL,
    "teacher_id": "fixture_only",
    "source_identity": identity,
    "trials": trials,
    "geometry": {},
  }
  (attempts / "MANIFEST.json").write_text(json.dumps(manifest))
  result = {"manifest_sha256": digest(manifest), "raw_sha256": hashes}
  (attempts / "RESULT.json").write_text(json.dumps(result))
  # A hand-written SCORE must be ignored; actual raw data decides admission.
  (attempts / "SCORE.json").write_text('{"qualified":false}')
  index = tmp_path / "index.json"
  index.write_text(json.dumps({"attempts": ["attempt"]}))
  result = require_qualification(index, "fixture_only", "student:flat_foundation")
  assert result["qualified"] and result["trials"] == len(trials)
  with pytest.raises(ValueError, match="不足"):
    require_qualification(index, "fixture_only", "student:flat_axes")
  first = attempts / trials[0]["raw_file"]
  first.write_text(
    first.read_text().replace('"native_failure": false', '"native_failure": true', 1)
  )
  with pytest.raises(ValueError, match="摘要"):
    require_qualification(index, "fixture_only", "student:flat_foundation")


def test_support_height_uses_contact_surface_and_ignores_airborne_wheel():
  from types import SimpleNamespace

  from wolf.rewards import support_height

  contact = SimpleNamespace(
    found=torch.tensor([[1, 1, 0, 0]]),
    force=torch.tensor(
      [[[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    ),
    pos=torch.tensor(
      [[[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    ),
  )
  env = SimpleNamespace(
    scene={
      "wheel_slip_contact": SimpleNamespace(data=contact),
      "robot": SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.9]]))
      ),
    }
  )
  torch.testing.assert_close(support_height(env), torch.zeros(1), atol=1e-12, rtol=0)
  contact.found.zero_()
  assert support_height(env).item() == 0.0


def test_weights_may_cross_runtime_but_full_resume_cannot(tmp_path, monkeypatch):
  from wolf.training import validate_resume

  original = {"sha256": "code_assets", "runtime": {"torch": "2.14.0+cu130"}}
  destination = {"sha256": "code_assets", "runtime": {"torch": "2.14.1+cu126"}}
  monkeypatch.setattr("wolf.checkpoint.source_identity", lambda: original)
  teacher, cnn, gru = initial_models(24)
  path = tmp_path / "teacher.pt"
  save_bundle(path, teacher, cnn, gru, phase="teacher", updates=1)
  monkeypatch.setattr("wolf.checkpoint.source_identity", lambda: destination)
  _, _, _, packet = load_bundle(path)
  assert packet["source_identity"] == original
  monkeypatch.setattr("wolf.training.source_identity", lambda: destination)
  with pytest.raises(ValueError, match="运行库"):
    validate_resume(packet, "teacher", {})
  monkeypatch.setattr(
    "wolf.checkpoint.source_identity",
    lambda: {**destination, "sha256": "different_assets"},
  )
  with pytest.raises(ValueError, match="源码或资产"):
    load_bundle(path)


def test_complete_resume_requires_same_identity_and_recipe(monkeypatch):
  from black.learner_state import SCHEMA
  from wolf.training import validate_resume

  identity = {"sha256": "same_code", "runtime": {"torch": "same_runtime"}}
  recipe = {"phase": "teacher", "max_level": 0}
  packet = {
    "phase": "teacher",
    "source_identity": identity,
    "completed_updates": 1,
    "learning_state": {"schema": SCHEMA, "recipe": recipe, "optimizer_calls": 20},
  }
  monkeypatch.setattr("wolf.training.source_identity", lambda: identity)
  assert validate_resume(packet, "teacher", recipe)["optimizer_calls"] == 20
  with pytest.raises(ValueError, match="recipe"):
    validate_resume(packet, "teacher", {**recipe, "max_level": 1})


def test_source_digest_excludes_runtime_and_tests(monkeypatch):
  from wolf.checkpoint import source_identity

  files = {
    "src/core.py": "original",
    "@runtime/versions.json": "runtime_a",
    "tests/test_fixture.py": "fixture_a",
  }
  runtime = {"torch": "runtime_a"}
  monkeypatch.setattr("black.eval_collect.source_files", lambda: dict(files))
  monkeypatch.setattr("black.eval_collect.runtime_identity", lambda: dict(runtime))
  first = source_identity()
  files["@runtime/versions.json"] = "runtime_b"
  files["tests/test_fixture.py"] = "fixture_b"
  runtime["torch"] = "runtime_b"
  second = source_identity()
  assert first["sha256"] == second["sha256"]
  assert first["runtime"] != second["runtime"]
  files["src/core.py"] = "modified"
  assert source_identity()["sha256"] != first["sha256"]
