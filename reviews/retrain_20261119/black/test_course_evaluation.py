"""综合/专项原始复评反例；合成通过记录不代表真实教师能力。"""

import json
import math
from types import SimpleNamespace

import pytest
import torch
from support.black.obstacle_geometry import GROUPS, historical_case
from support.black.parameters import PARAMETERS
from support.black.passage_course import POLICY
from support.black.specialty_env import make_specialty_configs

from black.course_evaluation import (
  SCHEMA,
  CourseAdapter,
  roster,
  score_attempt,
  score_trial,
)
from black.curriculum import configure, limit_student_specialty
from black.eval_collect import sha256
from black.test_eval_score import IDENTITY


def make_trial(spec):
  general = spec["suite"] == "general"
  if general:
    length = 3.61
    positions = [min((i + 1) * 0.005, length) for i in range(722)]
    heading = 0.0 if spec["direction"] > 0 else math.pi
  else:
    case = historical_case(spec["case"])[2]
    length = case["xmax"] - case["xmin"] + 1.5
    positions = [min((i + 1) * 0.1, length) for i in range(math.ceil(length / 0.1))]
    positions += [length] * 50
    heading = 0.0 if spec["direction"] > 0 else math.pi
    heading += spec["angle"] * math.pi / 180
  rows, previous = [], 0.0 if general or spec["direction"] > 0 else length
  for tick, progress in enumerate(positions):
    x = (
      progress * spec["direction"]
      if general
      else (progress if spec["direction"] > 0 else length - progress)
    )
    done = tick == len(positions) - 1
    row = dict(
      tick=tick,
      terminal_tick=tick + 1,
      episode=0,
      failed=False,
      depth_valid=True,
      frame=tick // 5,
      capture_tick=tick - tick % 5,
      post_episode=int(done),
      terrain_level=spec["level"],
      terrain_type=(
        1
        + ("slope_up", "slope_down", "stairs_up", "stairs_down", "rough", "wave").index(
          spec["domain"]
        )
        if general
        else 1
      ),
      pre_root=[previous, 0, 0.35],
      root=[x, 0, 0.35],
      pre_heading=heading,
      velocity=[0.25, 0, 0],
      command=[0.25, 0, 0],
      action=[0.0] * 12,
      actual_action=[0.0] * 12,
      linear_speed=0.0,
      angular_speed=0.0,
      tilt=0.0,
      done=done,
      terminated=done and not general,
      truncated=done and general,
    )
    if general:
      row["scan_valid"] = True
    else:
      row.update(
        foot_local=[[x, 0, 0]] * 4,
        direction=spec["direction"],
        angle=spec["angle"],
        contact=[True] * 4,
        lower_floor=[False] * 4,
        contact_invalid=False,
      )
    rows.append(row)
    previous = x
  return dict(spec=spec, rows=rows)


def save_attempt(directory, suite, seed, max_level=None):
  directory.mkdir()
  specs = roster(suite, seed, max_level)
  manifest = dict(
    suite=suite,
    base_seed=seed,
    max_level=max_level,
    roster=specs,
    max_controls=None,
    **IDENTITY,
  )
  # The common fixture supplies its own old schema; keep this collector's schema.
  manifest["schema"] = SCHEMA
  manifest_path, result_path = directory / "MANIFEST.json", directory / "RESULT.json"
  manifest_path.write_text(json.dumps(manifest))
  entries = []
  for spec in specs:
    trial = make_trial(spec)
    trial["manifest_sha256"] = sha256(manifest_path)
    path = directory / (spec["trial_id"] + ".json")
    path.write_text(json.dumps(trial))
    entries.append(dict(path=path.name, sha256=sha256(path)))
  result_path.write_text(
    json.dumps(
      dict(
        manifest_sha256=sha256(manifest_path), collection_complete=True, trials=entries
      )
    )
  )
  return manifest_path, result_path


@pytest.mark.parametrize(
  "suite", ["general", "platform", "high_wall", "slope", "broken_bridge", "plum_stakes"]
)
def test_complete_fixed_roster_and_raw_success(suite):
  specs = roster(suite, 191301, None if suite == "general" else 0)
  for spec in specs:
    assert score_trial(spec, make_trial(spec))
  expected = 48 if suite == "general" else 6
  assert len(specs) == expected
  assert {row["direction"] for row in specs} == {-1, 1}
  if suite != "general":
    assert len(roster(suite, 191301)) == len(GROUPS[suite]) * 6


@pytest.mark.parametrize(
  "bad",
  [
    "displacement",
    "domain",
    "command",
    "direction",
    "coverage",
    "premature_done",
    "post_reset",
    "action",
    "nan",
  ],
)
def test_general_rejects_false_training_success(bad):
  spec = roster("general", 191301)[1]
  trial = make_trial(spec)
  rows = trial["rows"]
  if bad == "displacement":
    for row in rows:
      row["root"] = row["pre_root"] = [0, 0, 0.35]
  elif bad == "domain":
    rows[10]["terrain_type"] = 0
  elif bad == "command":
    rows[10]["command"] = [0, 0, 0]
  elif bad == "direction":
    for row in rows:
      row["pre_heading"] += math.pi
  elif bad == "coverage":
    for row in rows[:100]:
      row["scan_valid"] = False
  elif bad == "premature_done":
    rows[10]["done"] = rows[10]["truncated"] = True
  elif bad == "post_reset":
    rows[10]["pre_root"] = [0, 0, 0.35]
  elif bad == "action":
    rows[10]["actual_action"][0] = 0.1
  else:
    rows[10]["velocity"][0] = float("nan")
  for row in rows:
    row.update(progress=1e6, demand=1, coverage=1, traversal=True)
  assert not score_trial(spec, trial)


@pytest.mark.parametrize(
  "bad",
  [
    "floor",
    "invalid_contact",
    "terminal_only",
    "support",
    "wrong_level",
    "wrong_direction",
    "wrong_angle",
    "no_entrance",
    "short_stop",
    "timeout",
    "outside",
  ],
)
def test_specialty_requires_actual_supported_full_route(bad):
  spec = roster("broken_bridge", 191301, 0)[3]
  trial = make_trial(spec)
  rows = trial["rows"]
  if bad == "floor":
    rows[5]["lower_floor"][0] = True
  elif bad == "invalid_contact":
    rows[5]["contact_invalid"] = True
  elif bad == "terminal_only":
    trial["rows"] = rows[-1:]
  elif bad == "support":
    rows[-1]["contact"][0] = False
  elif bad == "wrong_level":
    rows[5]["terrain_level"] += 1
  elif bad == "wrong_direction":
    rows[5]["direction"] *= -1
  elif bad == "wrong_angle":
    rows[5]["angle"] += 15
  elif bad == "no_entrance":
    rows[0]["pre_root"][0] = rows[-1]["root"][0]
  elif bad == "short_stop":
    for row in rows[-50:-20]:
      row["linear_speed"] = 0.2
  elif bad == "timeout":
    rows[-1].update(terminated=False, truncated=True)
  else:
    rows[5]["root"][1] = 10
  assert not score_trial(spec, trial)


def test_attempt_rejects_hash_change_identity_and_failed_marker(tmp_path):
  manifest, result = save_attempt(tmp_path / "attempt", "platform", 191301, 0)
  assert score_attempt(manifest, result)["counts"]["success"] == 6
  failed = result.parent / "FAILED.json"
  failed.write_text("{}")
  with pytest.raises(ValueError, match="failed attempt"):
    score_attempt(manifest, result)
  failed.unlink()
  trial = result.parent / "trial_0000.json"
  trial.write_text(trial.read_text() + " ")
  with pytest.raises(ValueError, match="哈希"):
    score_attempt(manifest, result)


def test_fixed_assignment_subset_and_snapshot_survive_reset():
  specs = roster("platform", 191301, 0)
  adapter = CourseAdapter(specs)
  n = len(specs)
  command = SimpleNamespace(
    flat_mask=torch.ones(n, dtype=torch.bool),
    route_direction=torch.zeros(n, dtype=torch.long),
    entry_angle=torch.zeros(n),
    route_sample_step=torch.zeros(n, dtype=torch.long),
  )
  terrain = SimpleNamespace(
    terrain_levels=torch.ones(n, dtype=torch.long),
    terrain_types=torch.zeros(n, dtype=torch.long),
    terrain_origins=torch.randn(3, 7, 3),
    env_origins=torch.zeros(n, 3),
  )
  env = SimpleNamespace(
    num_envs=n,
    device="cpu",
    common_step_counter=9,
    command_manager=SimpleNamespace(get_term=lambda _: command),
    scene=SimpleNamespace(terrain=terrain),
  )
  adapter.assign(env, torch.tensor([1, 4]))
  assert command.flat_mask.tolist() == [True, False, True, True, False, True]
  assert terrain.terrain_levels.tolist() == [1, 0, 1, 1, 0, 1]
  assert terrain.terrain_types.tolist() == [0, 1, 0, 0, 1, 0]
  torch.testing.assert_close(
    terrain.env_origins[[1, 4]], terrain.terrain_origins[0, 1].expand(2, -1)
  )
  adapter.armed = {}
  adapter.snapshot = dict(
    terminated=torch.tensor([True, False]),
    truncated=torch.tensor([False, True]),
    root=torch.tensor([[1.0, 0, 0], [2.0, 0, 0]]),
  )
  native = dict(
    critic=torch.zeros(2, 446)
  )  # reset observations do not replace terminal snapshot.
  adapter.finish(native, torch.tensor([True, True]))
  assert adapter.snapshot["root"][0, 0] == 1
  assert adapter.armed is None


def test_student_limits_specialty_levels_and_angles(monkeypatch):
  import support.black.specialty_env as specialty

  cfg, _ = make_specialty_configs("platform", num_envs=32, course_parameters=PARAMETERS)
  configure(cfg, "specialty")
  state = SimpleNamespace(height_level=5, angle_stage=3)
  monkeypatch.setattr(specialty, "specialty_state", lambda _: state)
  cfg.curriculum["specialty_course"].func = lambda env, ids: {}
  limit_student_specialty(cfg, 1)
  terrain = SimpleNamespace(
    terrain_levels=torch.tensor([0, 5]),
    terrain_types=torch.tensor([0, 1]),
    terrain_origins=torch.randn(12, 7, 3),
    env_origins=torch.zeros(2, 3),
  )
  cfg.curriculum["specialty_course"].func(
    SimpleNamespace(scene=SimpleNamespace(terrain=terrain)), None
  )
  assert state.height_level == 1 and state.angle_stage == 1
  assert terrain.terrain_levels.tolist() == [0, 1]
  cfg, _ = make_specialty_configs(
    "stairs", num_envs=32, course_parameters=PARAMETERS, passage_policy=POLICY
  )
  configure(cfg, "specialty")
  limit_student_specialty(cfg, 2)
  assert cfg.commands["twist"].passage_course_policy["maximum_cm"] == 8


def test_capture_clones_real_terminal_state_before_reset(monkeypatch):
  from mjlab.tasks.velocity.config.black_rough import mdp

  class Scene(dict):
    env_origins: torch.Tensor
    terrain: SimpleNamespace

  n = 2
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[3.61, 0, 0.3], [-3.61, 0, 0.3]]),
      root_link_lin_vel_b=torch.zeros(n, 3),
      root_link_lin_vel_w=torch.zeros(n, 3),
      root_link_ang_vel_w=torch.zeros(n, 3),
      projected_gravity_b=torch.tensor([[0.0, 0, -1]] * n),
      heading_w=torch.zeros(n),
    )
  )
  scene = Scene(
    robot=robot,
    contacts=SimpleNamespace(data=SimpleNamespace(found=torch.ones(n, 4))),
    foot_height_scan=SimpleNamespace(
      data=SimpleNamespace(distances=torch.ones(n, 4, 1), heights=torch.zeros(n, 4))
    ),
  )
  scene.env_origins = torch.zeros(n, 3)
  scene.terrain = SimpleNamespace(
    terrain_levels=torch.tensor([1, 1]), terrain_types=torch.tensor([1, 1])
  )
  command = SimpleNamespace(command=torch.tensor([[0.25, 0, 0]] * n))
  failed = torch.tensor([False, False])
  monkeypatch.setattr(mdp, "state", lambda _: SimpleNamespace(failed=failed))
  plant = SimpleNamespace(
    num_envs=n,
    common_step_counter=1,
    scene=scene,
    command_manager=SimpleNamespace(
      get_term=lambda _: command, get_command=lambda _: command.command
    ),
    cfg=SimpleNamespace(
      observations={
        "actor": SimpleNamespace(
          terms={"foot_contact": SimpleNamespace(params={"sensor_name": "contacts"})}
        )
      }
    ),
    reward_manager=SimpleNamespace(compute=lambda: 3),
    reset_terminated=torch.tensor([False, False]),
    reset_time_outs=torch.tensor([True, True]),
    action_manager=SimpleNamespace(action=torch.zeros(n, 12)),
  )
  env = SimpleNamespace(plant=plant)
  adapter = CourseAdapter(roster("general", 191301)[:n])
  adapter.attach(env)
  history = torch.zeros(n, 5, 45)
  history[:, -1, 30:33] = command.command
  adapter.begin({"history": history})
  assert plant.reward_manager.compute() == 3
  # These are the same mutable native buffers which a reset would overwrite.
  robot.data.root_link_pos_w.zero_()
  plant.reset_time_outs.zero_()
  failed.fill_(True)
  command.command.zero_()
  adapter.finish(
    {"history": torch.zeros_like(history)}, torch.ones(n, dtype=torch.bool)
  )
  assert adapter.snapshot is not None
  assert adapter.snapshot["root"][0, 0].item() == pytest.approx(3.61)
  assert adapter.snapshot["truncated"].all()
  assert not adapter.snapshot["failed"].any()
  assert adapter.snapshot["command"][0, 0] == 0.25
  adapter.detach()


def test_actual_spawn_heading_must_match_registered_orientation():
  for suite in ("general", "platform"):
    spec = roster(suite, 191301, None if suite == "general" else 0)[0]
    trial = make_trial(spec)
    trial["rows"][0]["pre_heading"] += 0.1
    assert not score_trial(spec, trial)
