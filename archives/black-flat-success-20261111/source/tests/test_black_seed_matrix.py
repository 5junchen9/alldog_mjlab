"""No PPO updates, simulator creation or CUDA work in this acceptance suite."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mjlab.scripts.black_seed_matrix import (
  REWARD_NAMES,
  TEMPLATE,
  atomic_json,
  execute_plan,
  file_sha,
  guard_resources,
  install_plan,
  lock,
  main,
  make_plan,
  validate_config,
)
from mjlab.scripts.train_black_seed import checkpoint_records


@pytest.fixture
def cfg():
  value = copy.deepcopy(TEMPLATE)
  value.update(
    experiment_id="test_only_2x3",
    selection_rule="Lowest mean command MAE subject to zero falls; ties earlier update",
    selection_seeds=[11001, 11002],
    final_seeds=[21001, 21002],
    configs=[
      {"id": "baseline", "train_seeds": [11, 12, 13], "overrides": {}},
      {
        "id": "lr_example",
        "train_seeds": [21, 22, 23],
        "overrides": {"learning_rate": 0.0003},
      },
    ],
  )
  return value


def test_rectangular_real_seed_matrix(cfg, tmp_path):
  plan = make_plan(cfg, tmp_path, {"source.py": "test"})
  assert (plan["m"], plan["n"], len(plan["jobs"])) == (2, 3, 6)
  assert len({job["directory"] for job in plan["jobs"]}) == 6
  assert len({job["training_seed"] for job in plan["jobs"]}) == 6
  assert cfg["num_envs"] == 4096
  assert plan["total_independent_trainings"] == 6
  assert plan["parallel_jobs"] == 1
  assert list(tmp_path.iterdir()) == []


def test_template_needs_decision():
  with pytest.raises(ValueError):
    validate_config(TEMPLATE)


@pytest.mark.parametrize(
  "field,value",
  [
    ("iterations", 1999),
    ("iterations", True),
    ("num_envs", 0),
    ("selection_seeds", []),
    ("final_seeds", [11001]),
    ("final_seeds", [42, 42]),
    ("selection_rule", ""),
    ("device", "cuda:0,1"),
    ("eval_pushes", "yes"),
    ("min_free_disk_gb", 0),
  ],
)
def test_bad_config_rejected(cfg, field, value):
  cfg[field] = value
  with pytest.raises(ValueError):
    validate_config(cfg)


@pytest.mark.parametrize("seed", [11, 11001, 21001, -1, 2**31, True])
def test_seed_domains_rejected(cfg, seed):
  cfg["configs"][1]["train_seeds"][0] = seed
  with pytest.raises(ValueError):
    validate_config(cfg)


def test_ragged_matrix_rejected(cfg):
  cfg["configs"][1]["train_seeds"].pop()
  with pytest.raises(ValueError, match="same n"):
    validate_config(cfg)


@pytest.mark.parametrize(
  "overrides",
  [
    {"resume": True},
    {"learning_rate": 0},
    {"entropy_coef": -1},
    {"learning_rate": float("nan")},
    {"reward_weights": {"typo": 1}},
  ],
)
def test_override_guards(cfg, overrides):
  cfg["configs"][0]["overrides"] = overrides
  with pytest.raises(ValueError):
    validate_config(cfg)


def test_actual_task_bindings():
  from mjlab.rl import RslRlOnPolicyRunnerCfg
  from mjlab.scripts.black_seed_matrix import TASK
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  env, agent = load_env_cfg(TASK), load_rl_cfg(TASK)
  assert isinstance(agent, RslRlOnPolicyRunnerCfg)
  assert set(env.rewards) == REWARD_NAMES
  assert hasattr(agent.algorithm, "learning_rate")
  assert hasattr(agent.algorithm, "entropy_coef")
  assert agent.resume is False
  env.seed = 999
  assert load_env_cfg(TASK).seed != 999


def test_default_cli_never_launches(cfg, tmp_path, monkeypatch, capsys):
  config = tmp_path / "config.json"
  config.write_text(json.dumps(cfg))
  output = tmp_path / "not_created"
  monkeypatch.setattr(
    "sys.argv", ["matrix", "--config", str(config), "--output", str(output)]
  )
  with (
    patch("mjlab.scripts.black_seed_matrix.source_manifest", return_value={}),
    patch(
      "mjlab.scripts.black_seed_matrix.run_child",
      side_effect=AssertionError("launched"),
    ),
  ):
    main()
  assert "DRY RUN" in capsys.readouterr().err
  assert not output.exists()


def test_plan_cannot_change(cfg, tmp_path):
  plan = make_plan(cfg, tmp_path, {})
  path = tmp_path / "plan.json"
  install_plan(path, plan)
  install_plan(path, plan)
  cfg["iterations"] = 3000
  with pytest.raises(ValueError, match="differs"):
    install_plan(path, make_plan(cfg, tmp_path, {}))


def test_lock_exclusion(tmp_path):
  with lock(tmp_path / "one.lock"), pytest.raises(RuntimeError, match="holds"):
    with lock(tmp_path / "one.lock"):
      pass


def test_busy_gpu_refused(cfg, tmp_path):
  with patch(
    "mjlab.scripts.black_seed_matrix.subprocess.check_output", return_value="123\n"
  ):
    with pytest.raises(RuntimeError, match="compute processes"):
      guard_resources(cfg, tmp_path)


def test_gpu_inspection_failure_is_not_bypassed(cfg, tmp_path):
  with patch(
    "mjlab.scripts.black_seed_matrix.subprocess.check_output", side_effect=OSError
  ):
    with pytest.raises(OSError):
      guard_resources(cfg, tmp_path)


def test_complete_cells_skip_without_child(cfg, tmp_path):
  plan = make_plan(cfg, tmp_path, {})
  for job in plan["jobs"]:
    directory = Path(job["directory"])
    checkpoint = directory / "attempts" / "one" / "model_1999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint fixture")
    record = {
      "fingerprint": plan["fingerprint"],
      "run_id": job["run_id"],
      "training_seed": job["training_seed"],
      "completed_updates": 2000,
      "sha256": file_sha(checkpoint),
    }
    atomic_json(checkpoint.with_suffix(".pt.json"), record)
    atomic_json(
      directory / "status.json",
      {
        "state": "completed",
        "completed_updates": 2000,
        "checkpoint": {**record, "path": str(checkpoint)},
      },
    )
  with (
    patch("mjlab.scripts.black_seed_matrix.guard_resources"),
    patch("mjlab.scripts.black_seed_matrix.source_manifest", return_value={}),
    patch(
      "mjlab.scripts.black_seed_matrix.run_child",
      side_effect=AssertionError("launched"),
    ),
  ):
    execute_plan(plan, resume=True)


def test_checkpoint_integrity_and_numeric_order(tmp_path):
  attempt = tmp_path / "attempts" / "one"
  attempt.mkdir(parents=True)
  for i in (9, 99, 100):
    path = attempt / f"model_{i}.pt"
    path.write_bytes(f"checkpoint fixture {i}".encode())
    atomic_json(
      path.with_suffix(".pt.json"),
      {
        "run_id": "a/seed_11",
        "fingerprint": "plan",
        "completed_updates": i + 1,
        "sha256": file_sha(path),
      },
    )
  (attempt / "model_100.pt").write_bytes(b"incomplete or corrupted")
  records = checkpoint_records(tmp_path, "plan", "a/seed_11")
  assert [record["completed_updates"] for record in records] == [10, 100]
  with pytest.raises(ValueError, match="another run"):
    checkpoint_records(tmp_path, "plan", "other/seed_12")


def test_completed_status_cannot_hide_corruption(cfg, tmp_path):
  from mjlab.scripts.black_seed_matrix import verify_completed

  plan = make_plan(cfg, tmp_path, {})
  job = plan["jobs"][0]
  directory = Path(job["directory"])
  atomic_json(directory / "status.json", {"state": "completed"})
  with pytest.raises((ValueError, KeyError, FileNotFoundError)):
    verify_completed(plan, job)


def test_fresh_process_seed_and_distributed_isolation(tmp_path, monkeypatch):
  from mjlab.scripts.black_seed_matrix import run_child

  monkeypatch.setenv("WORLD_SIZE", "8")
  monkeypatch.setenv("RANK", "3")
  monkeypatch.setenv("LOCAL_RANK", "3")
  with patch("mjlab.scripts.black_seed_matrix.subprocess.Popen") as popen:
    popen.return_value.wait.return_value = 0
    assert run_child(["do-not-execute"], tmp_path / "console.log", 1, 123) == 0
  env = popen.call_args.kwargs["env"]
  assert env["PYTHONHASHSEED"] == "123"
  assert not {"WORLD_SIZE", "RANK", "LOCAL_RANK"} & env.keys()
  assert popen.call_args.kwargs["start_new_session"] is True


def test_command_observations_bypass_cache():
  from types import SimpleNamespace
  from unittest.mock import Mock

  import torch

  from mjlab.scripts.evaluate_black_seed import current_observations

  manager = Mock()
  manager.compute_group.return_value = torch.ones((2, 72))
  env = SimpleNamespace(
    observation_manager=manager,
    num_envs=2,
    cfg=SimpleNamespace(observations={"actor": None, "critic": None}),
  )
  result = current_observations(env)
  assert result["actor"].shape == (2, 72)
  assert manager.compute_group.call_count == 2
  manager.compute.assert_not_called()
  manager.compute_group.assert_any_call("actor", update_history=False)


def test_teacher_observations_support_uncached_evaluation():
  from mjlab.scripts.black_seed_matrix import TASK
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK)
  for group in cfg.observations.values():
    assert group.history_length in (None, 0)
    assert not group.enable_corruption
    for term in group.terms.values():
      assert term.history_length == 0
      assert term.delay_max_lag == 0


def test_final_evaluation_requires_frozen_selection(cfg, tmp_path):
  from mjlab.scripts.evaluate_black_seed import run_evaluation

  plan = make_plan(cfg, tmp_path, {})
  with (
    patch("mjlab.scripts.evaluate_black_seed.guard_resources"),
    pytest.raises(FileNotFoundError),
  ):
    run_evaluation(plan, plan["jobs"][0], {}, "final")


def test_selection_rejected_after_freeze(cfg, tmp_path):
  from mjlab.scripts.evaluate_black_seed import run_evaluation

  plan = make_plan(cfg, tmp_path, {})
  atomic_json(tmp_path / "selection_frozen.json", {})
  with (
    patch("mjlab.scripts.evaluate_black_seed.guard_resources"),
    pytest.raises(ValueError, match="frozen"),
  ):
    run_evaluation(plan, plan["jobs"][0], {}, "selection")


def test_freeze_requires_all_training_seeds(cfg, tmp_path):
  from mjlab.scripts.evaluate_black_seed import freeze_selection

  plan = make_plan(cfg, tmp_path, {})
  mapping = tmp_path / "mapping.json"
  atomic_json(mapping, {plan["jobs"][0]["run_id"]: "some.pt"})
  with pytest.raises(ValueError, match="EVERY"):
    freeze_selection(plan, mapping, execute=False)


def test_final_rejects_different_checkpoint(cfg, tmp_path):
  from mjlab.scripts.evaluate_black_seed import run_evaluation

  plan = make_plan(cfg, tmp_path, {})
  job = plan["jobs"][0]
  atomic_json(
    tmp_path / "selection_frozen.json",
    {
      "fingerprint": plan["fingerprint"],
      "entries": {job["run_id"]: {"checkpoint": {"path": "fixed"}}},
    },
  )
  with (
    patch("mjlab.scripts.evaluate_black_seed.guard_resources"),
    pytest.raises(ValueError, match="frozen selection"),
  ):
    run_evaluation(plan, job, {"path": "changed"}, "final")


@pytest.mark.parametrize(
  "last_iteration,completed", [(0, 1), (999, 1000), (1999, 2000)]
)
def test_resume_uses_next_index_not_last_index(last_iteration, completed):
  from mjlab.scripts.train_black_seed import resume_start

  identity = {
    "run_id": "a/seed_11",
    "fingerprint": "plan",
    "training_seed": 11,
    "runtime": {},
  }
  record = {"completed_updates": completed}
  info = {"black_matrix": {**identity, **record}}
  assert resume_start(record, info, last_iteration, identity) == completed
  assert 2000 - resume_start(record, info, last_iteration, identity) == 2000 - completed
  with pytest.raises(ValueError):
    resume_start(record, info, last_iteration + 1, identity)


def test_resume_rejects_cross_seed_metadata():
  from mjlab.scripts.train_black_seed import resume_start

  expected = {
    "run_id": "a/seed_11",
    "fingerprint": "plan",
    "training_seed": 11,
    "runtime": {},
  }
  record = {"completed_updates": 1000}
  info = {"black_matrix": {**expected, **record, "training_seed": 12}}
  with pytest.raises(ValueError):
    resume_start(record, info, 999, expected)


@pytest.mark.parametrize("profile", ["LowSpeed", "HighSpeed"])
def test_speed_projects_require_explicit_bounds(cfg, profile):
  cfg["configs"][0]["task"] = f"Mjlab-Velocity-Flat-Black-Teacher-{profile}"
  with pytest.raises(ValueError, match="explicit"):
    validate_config(cfg)


@pytest.mark.parametrize(
  "speed",
  [
    {"min_speeds": None, "max_speeds": None},
    {"min_speeds": [0.05, 0.05, 0.05], "max_speeds": [0.02, 1, 1]},
    {"min_speeds": [0, 0.05, 0.05], "max_speeds": [1, 1, 1]},
  ],
)
def test_speed_bounds_fail_closed(cfg, speed):
  cfg["configs"][0]["task"] = "Mjlab-Velocity-Flat-Black-Teacher-LowSpeed"
  cfg["configs"][0]["overrides"] = {"speed": speed}
  with pytest.raises(ValueError):
    validate_config(cfg)


def test_original_omni_cannot_be_changed_into_speed_project(cfg):
  cfg["configs"][0]["overrides"] = {
    "speed": {"min_speeds": [0.05] * 3, "max_speeds": [0.5] * 3}
  }
  with pytest.raises(ValueError, match="preserved"):
    validate_config(cfg)


@pytest.mark.parametrize("profile", ["LowSpeed", "HighSpeed"])
def test_speed_project_matrix_bindings(cfg, tmp_path, profile):
  from mjlab.scripts.black_seed_matrix import apply_overrides
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.tasks.velocity.config.black_teacher.speed_command import (
    ConfiguredOmniVelocityCommandCfg,
  )

  task = f"Mjlab-Velocity-Flat-Black-Teacher-{profile}"
  cfg["configs"][0]["task"] = task
  # Test fixtures only, not the user's approved speed parameters.
  cfg["configs"][0]["overrides"] = {
    "speed": {"min_speeds": [0.02, 0.03, 0.04], "max_speeds": [0.1, 0.2, 0.3]}
  }
  plan = make_plan(cfg, tmp_path, {})
  job = plan["jobs"][0]
  assert job["task"] == task
  env, agent = load_env_cfg(task), load_rl_cfg(task)
  before = {name: term.weight for name, term in env.rewards.items()}
  apply_overrides(env, agent, job)
  assert isinstance(env.commands["twist"], ConfiguredOmniVelocityCommandCfg)
  assert env.commands["twist"].min_speeds == (0.02, 0.03, 0.04)
  assert env.commands["twist"].ranges.lin_vel_y == (-0.2, 0.2)
  assert before == {name: term.weight for name, term in env.rewards.items()}


def test_speed_evaluation_uses_project_bounds():
  from mjlab.scripts.evaluate_black_seed import scenarios

  rows = scenarios({"min_speeds": [0.02, 0.03, 0.04], "max_speeds": [0.1, 0.2, 0.3]})
  assert any(row["command"] == [0.02, 0, 0] for row in rows)
  assert all(abs(row["command"][0]) <= 0.1 for row in rows)
  assert all(abs(row["command"][1]) <= 0.2 for row in rows)
  assert all(abs(row["command"][2]) <= 0.3 for row in rows)
  assert len(scenarios()) == 35
