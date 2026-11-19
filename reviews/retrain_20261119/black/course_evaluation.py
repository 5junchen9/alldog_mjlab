"""综合地形与五项非楼梯专项的固定面板、原始采集和独立复评。"""

import argparse
import copy
import fcntl
import json
import math
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from support.black.obstacle_geometry import GROUPS, historical_case

from black.checkpoint import load_bundle, teacher_identity
from black.eval_collect import device_lock_path, sha256, source_files, write_json
from black.eval_protocol import EVALUATION_SEEDS, digest
from black.eval_score import require
from black.runtime import NativeEnv

SCHEMA = "black_retrain_course_evaluation_v1"
SUITES = ("general", "platform", "high_wall", "slope", "broken_bridge", "plum_stakes")
DOMAINS = ("slope_up", "slope_down", "stairs_up", "stairs_down", "rough", "wave")


def roster(suite, seed, max_level=None) -> list[dict]:
  rows = []
  if suite == "general":
    if max_level is not None:
      raise ValueError("综合地形使用完整四档，不能截取难度")
    for domain in DOMAINS:
      for level in (1, 2, 3, 4):
        for direction in (-1, 1):
          rows.append(
            dict(domain=domain, level=level, direction=direction, horizon=1800)
          )
  else:
    if suite not in SUITES:
      raise ValueError("此入口只采集综合地形及五项非楼梯专项")
    maximum = len(GROUPS[suite]) - 1 if max_level is None else max_level
    if not 0 <= maximum < len(GROUPS[suite]):
      raise ValueError("专项难度上限超出已定义的几何")
    for level, case in enumerate(GROUPS[suite][: maximum + 1]):
      for direction in (-1, 1):
        for angle in (-15, 0, 15):
          rows.append(
            dict(case=case, level=level, direction=direction, angle=angle, horizon=6000)
          )
  for i, row in enumerate(rows):
    row.update(
      trial_id=f"trial_{i:04d}",
      suite=suite,
      batch_id=i // 32,
      lane=i % 32,
      seed=seed + i // 32,
    )
  return rows


class CourseAdapter:
  def __init__(self, rows):
    self.rows = rows
    self.source_course = "general" if rows[0]["suite"] == "general" else "specialty"
    self.env: NativeEnv | None = None
    self.snapshot = None
    self.armed = None

  def assign(self, env, ids):
    ids = torch.arange(env.num_envs, device=env.device) if ids is None else ids
    command = env.command_manager.get_term("twist")
    command.flat_mask[ids] = False
    terrain = env.scene.terrain
    levels = torch.tensor([row["level"] for row in self.rows], device=env.device)
    if self.source_course == "general":
      names = list(env.cfg.scene.terrain.terrain_generator.sub_terrains)
      kinds = [names.index(row["domain"]) for row in self.rows]
    else:
      kinds = [1] * env.num_envs
      command.route_direction[ids] = torch.tensor(
        [row["direction"] for row in self.rows], device=env.device
      )[ids]
      command.entry_angle[ids] = torch.tensor(
        [row["angle"] * math.pi / 180 for row in self.rows], device=env.device
      )[ids]
      command.route_sample_step[ids] = env.common_step_counter
    terrain.terrain_levels[ids] = levels[ids]
    terrain.terrain_types[ids] = torch.tensor(kinds, device=env.device)[ids]
    terrain.env_origins[ids] = terrain.terrain_origins[
      terrain.terrain_levels[ids], terrain.terrain_types[ids]
    ]
    return {"fixed_evaluation_roster": True}

  def configure(self, cfg):
    from support.black.specialty_composition import CompositionCommand
    from support.black.specialty_env import SpecialtyCommand

    from mjlab.managers import CurriculumTermCfg

    adapter = self
    parent = CompositionCommand if self.source_course == "general" else SpecialtyCommand

    class FixedCommand(parent):  # ty: ignore[unsupported-base] -- two concrete native command classes
      def publish(self):
        if adapter.source_course == "general":
          self.vel_command_b.zero_()
          self.vel_command_b[:, 0] = 0.25

      def compute(self, dt, env_ids=None):
        result = super().compute(dt, env_ids)
        self.publish()
        return result

      def reset(self, env_ids=None):
        result = super().reset(env_ids)
        self.publish()
        return result

      def sample_route_entries(self, env_ids):
        adapter.assign(self._env, env_ids)

    original = cfg.commands["twist"]

    class FixedCommandCfg(type(original)):
      def build(self, env):
        return FixedCommand(self, env)

    cfg.commands["twist"] = FixedCommandCfg(
      **{f.name: copy.deepcopy(getattr(original, f.name)) for f in fields(original)}
    )
    cfg.curriculum = {"fixed_evaluation": CurriculumTermCfg(func=self.assign)}
    original_reset = cfg.events["reset_base"].func

    def reset(env, env_ids, pose_range, velocity_range):
      adapter.assign(env, env_ids)
      if adapter.source_course == "specialty":
        return original_reset(env, env_ids, pose_range, velocity_range)
      ids = (
        torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
      )
      for sign in (-1, 1):
        selected = [i for i in ids.tolist() if adapter.rows[i]["direction"] == sign]
        if selected:
          ranges = dict(pose_range)
          yaw = 0.0 if sign > 0 else math.pi
          ranges["yaw"] = (yaw, yaw)
          original_reset(
            env, torch.tensor(selected, device=env.device), ranges, velocity_range
          )

    cfg.events["reset_base"].func = reset

  def attach(self, env):
    self.env = env
    self.original_compute = env.plant.reward_manager.compute

    def compute(*args, **kwargs):
      result = self.original_compute(*args, **kwargs)
      self.capture()
      return result

    env.plant.reward_manager.compute = compute

  def capture(self):
    from support.black.specialty_env import SpecialtyCommand, specialty_state

    from mjlab.tasks.velocity.config.black_rough.mdp import state

    assert self.env is not None
    env = self.env.plant
    robot = env.scene["robot"]
    command = env.command_manager.get_term("twist")
    terrain = env.scene.terrain
    assert command is not None and terrain is not None
    params = env.cfg.observations["actor"].terms["foot_contact"].params
    contact = env.scene[params["sensor_name"]].data.found.reshape(env.num_envs, 4) > 0
    assert self.armed is not None
    if not torch.equal(command.command.detach().cpu(), self.armed["command"]):
      raise RuntimeError("动作前命令与终止前评分命令不同帧")
    values = dict(
      **self.armed,
      terminal_tick=torch.full((env.num_envs,), env.common_step_counter),
      terrain_level=terrain.terrain_levels,
      terrain_type=terrain.terrain_types,
      terminated=env.reset_terminated,
      truncated=env.reset_time_outs,
      root=robot.data.root_link_pos_w - env.scene.env_origins,
      velocity=robot.data.root_link_lin_vel_b,
      linear_speed=robot.data.root_link_lin_vel_w.norm(dim=-1),
      angular_speed=robot.data.root_link_ang_vel_w.norm(dim=-1),
      tilt=torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)),
      contact=contact,
      actual_action=env.action_manager.action,
    )
    if self.source_course == "general":
      st = state(env)
      scan = env.scene["foot_height_scan"].data
      foot_valid = (scan.distances.reshape(env.num_envs, 4, -1) >= 0).any(-1)
      values.update(
        failed=st.failed,
        scan_valid=foot_valid.all(-1) & torch.isfinite(scan.heights).all(-1),
      )
    else:
      assert isinstance(command, SpecialtyCommand)
      st = specialty_state(env)
      positions = robot.data.site_pos_w[:, st.feet] - env.scene.env_origins[:, None, :]
      sparse = (
        st.sparse_catalog.read(env.sim.data, env.num_envs)
        if st.sparse_catalog is not None
        else None
      )
      values.update(
        failed=st.failed,
        foot_local=positions,
        direction=command.route_direction,
        angle=command.entry_angle * 180 / math.pi,
        contact=sparse["supported"] if sparse is not None else contact,
        lower_floor=sparse["lower_floor"]
        if sparse is not None
        else torch.zeros_like(contact),
        contact_invalid=sparse["invalid"]
        if sparse is not None
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
      )
    self.snapshot = {
      name: value.detach().cpu().clone() for name, value in values.items()
    }

  def begin(self, native):
    if self.armed is not None:
      raise RuntimeError("上一评估动作尚未闭合")
    assert self.env is not None
    env = self.env.plant
    command = env.command_manager.get_command("twist")
    assert isinstance(command, torch.Tensor)
    current = native["history"].reshape(env.num_envs, 5, 45)[:, -1, 30:33]
    if not torch.equal(current, command):
      raise RuntimeError("动作输入与当前命令不同帧")
    robot = env.scene["robot"]
    values = dict(
      command=command,
      pre_root=robot.data.root_link_pos_w - env.scene.env_origins,
      pre_heading=robot.data.heading_w,
    )
    self.armed = {name: value.detach().cpu().clone() for name, value in values.items()}
    self.snapshot = None

  def finish(self, native, done):
    if self.snapshot is None or self.armed is None:
      raise RuntimeError("缺少终止前原始快照")
    terminal = self.snapshot["terminated"] | self.snapshot["truncated"]
    if not torch.equal(terminal, done.detach().cpu().bool()):
      raise RuntimeError("终止前快照与实际终止不一致")
    self.armed = None

  def detach(self):
    if self.env is not None:
      self.env.plant.reward_manager.compute = self.original_compute
      self.env = None


def score_trial(spec, trial):
  """从逐步位移、足位置和实际接触复算；不使用训练累计分或success。"""
  rows = trial["rows"]
  if not rows or trial["spec"] != spec or len(rows) > spec["horizon"]:
    return False
  streak, entry_seen = 0, False
  progress, demand, coverage = 0.0, 0.0, 0
  previous = None
  if spec["suite"] != "general":
    case = historical_case(spec["case"])[2]
    length = case["xmax"] - case["xmin"] + 1.5
    width = case["half_width"]
  for tick, row in enumerate(rows):
    if (
      row["tick"] != tick
      or row["terminal_tick"] != tick + 1
      or row["episode"] != 0
      or row["failed"] is not False
      or row["depth_valid"] is not True
      or not 0 <= row["capture_tick"] <= tick
      or tick - row["capture_tick"] >= 5
      or row["frame"] < 0
      or row["post_episode"] != int(row["done"])
      or row["terrain_level"] != spec["level"]
      or row["done"] != (row["terminated"] or row["truncated"])
      or (tick < len(rows) - 1 and row["done"])
    ):
      return False
    arrays = {
      name: row[name]
      for name in ("pre_root", "root", "velocity", "command", "action", "actual_action")
    }
    if any(
      len(value) != (12 if "action" in name else 3) for name, value in arrays.items()
    ):
      return False
    numeric = [value for array in arrays.values() for value in array]
    numeric += [
      row["pre_heading"],
      row["tilt"],
      row["linear_speed"],
      row["angular_speed"],
    ]
    if not all(math.isfinite(value) for value in numeric):
      return False
    if tick == 0:
      heading = 0.0 if spec["direction"] > 0 else math.pi
      if spec["suite"] != "general":
        heading += spec["angle"] * math.pi / 180
      error = row["pre_heading"] - heading
      if abs(math.atan2(math.sin(error), math.cos(error))) > 1e-4:
        return False
    if row["actual_action"] != row["action"]:
      return False
    if previous is not None and math.dist(row["pre_root"], previous) > 1e-4:
      return False
    if math.dist(row["root"], row["pre_root"]) > 0.25:
      return False
    previous = row["root"]
    if spec["suite"] == "general":
      # Native rough columns follow this explicit order.
      if row["terrain_type"] != DOMAINS.index(spec["domain"]) + 1:
        return False
      if row["command"] != [0.25, 0.0, 0.0]:
        return False
      if tick == 0 and (max(abs(v) for v in row["pre_root"][:2]) > 0.21):
        return False
      delta = [row["root"][i] - row["pre_root"][i] for i in (0, 1)]
      # The first native control suppresses reset displacement in its accumulator.
      if tick:
        progress += delta[0] * math.cos(row["pre_heading"]) + delta[1] * math.sin(
          row["pre_heading"]
        )
      demand += 0.25 * 0.02
      coverage += row["scan_valid"] is True
    else:
      foot = row["foot_local"]
      if len(foot) != 4 or any(len(position) != 3 for position in foot):
        return False
      if not all(math.isfinite(v) for position in foot for v in position):
        return False
      if (
        row["terrain_type"] != 1
        or row["direction"] != spec["direction"]
        or not math.isfinite(row["angle"])
        or abs(row["angle"] - spec["angle"]) > 1e-4
        or row["contact_invalid"]
        or any(row["lower_floor"])
      ):
        return False
      if (
        len(row["contact"]) != 4
        or len(row["lower_floor"]) != 4
        or any(type(value) is not bool for value in row["contact"] + row["lower_floor"])
      ):
        return False
      foot_progress = [p[0] if spec["direction"] > 0 else length - p[0] for p in foot]
      root_progress = (
        row["root"][0] if spec["direction"] > 0 else length - row["root"][0]
      )
      if abs(row["root"][1]) > width or not -0.70 <= root_progress <= length + 0.70:
        return False
      if tick == 0:
        initial = (
          row["pre_root"][0] if spec["direction"] > 0 else length - row["pre_root"][0]
        )
        entry_seen = abs(initial) < 0.25 and min(foot_progress) < 0.75
      stable = (
        min(foot_progress) >= length - 0.75
        and max(foot_progress) <= length + 0.70
        and max(abs(p[1]) for p in foot) <= width
        and max(abs(p[2]) for p in foot) <= 0.08
        and all(v is True for v in row["contact"])
        and row["linear_speed"] <= 0.08
        and row["angular_speed"] <= 0.12
        and row["tilt"] <= 0.35
      )
      streak = streak + 1 if stable else 0
  last = rows[-1]
  if spec["suite"] == "general":
    heading = last["pre_heading"]
    intended = last["root"][0] * math.cos(heading) + last["root"][1] * math.sin(heading)
    return bool(
      last["done"]
      and not last["terminated"]
      and max(abs(v) for v in last["root"][:2]) > 3.6
      and intended > 0
      and demand >= 1
      and progress >= 0.8 * demand
      and coverage / len(rows) >= 0.95
    )
  return bool(entry_seen and streak >= 50 and last["done"] and last["terminated"])


def score_attempt(manifest_path, result_path):
  manifest_path, result_path = Path(manifest_path), Path(result_path)
  require(not (result_path.parent / "FAILED.json").exists(), "failed attempt")
  manifest = json.loads(manifest_path.read_text())
  result = json.loads(result_path.read_text())
  if manifest["schema"] != SCHEMA or result["manifest_sha256"] != sha256(manifest_path):
    raise ValueError("评估身份或manifest哈希不符")
  expected = roster(manifest["suite"], manifest["base_seed"], manifest["max_level"])
  if manifest["roster"] != expected or len(result["trials"]) != len(expected):
    raise ValueError("固定面板缺失或被替换")
  scored = []
  for spec, item in zip(expected, result["trials"], strict=True):
    path = result_path.parent / item["path"]
    require(
      path.parent.resolve() == result_path.parent.resolve(),
      "trial path escapes attempt",
    )
    if sha256(path) != item["sha256"]:
      raise ValueError("原始轨迹哈希不符")
    trial = json.loads(path.read_text())
    require(
      trial["manifest_sha256"] == sha256(manifest_path), "trial identity mismatch"
    )
    scored.append(
      dict(
        trial_id=spec["trial_id"],
        success=score_trial(spec, trial),
        invalid_depth_frames=sum(not row["depth_valid"] for row in trial["rows"]),
      )
    )
  return dict(
    collection_complete=result["collection_complete"],
    bundle_sha256=manifest["bundle_sha256"],
    manifest_sha256=sha256(manifest_path),
    result_sha256=sha256(result_path),
    counts=dict(
      assigned=len(expected),
      success=sum(row["success"] for row in scored),
      independent_episodes=sum(
        bool(json.loads((result_path.parent / item["path"]).read_text())["rows"])
        for item in result["trials"]
      ),
    ),
    trials=scored,
  )


def _execute(args):
  rows = roster(args.suite, args.seed, args.max_level)
  source = source_files()
  teacher, _, _, _ = load_bundle(args.bundle, args.device)
  teacher.freeze()
  manifest: dict = dict(
    schema=SCHEMA,
    suite=args.suite,
    base_seed=args.seed,
    max_level=args.max_level,
    roster=rows,
    max_controls=args.max_controls,
    teacher_id=teacher_identity(teacher),
    bundle_sha256=sha256(args.bundle),
    source_sha256=digest(source),
    source_files=source,
  )
  write_json(args.out / "MANIFEST.json", manifest)
  trials: list[dict] = [
    dict(spec=row, rows=[], manifest_sha256=sha256(args.out / "MANIFEST.json"))
    for row in rows
  ]
  for batch in range(1 + rows[-1]["batch_id"]):
    selected = [trial for trial in trials if trial["spec"]["batch_id"] == batch]
    padded = [trial["spec"] for trial in selected]
    padded += [copy.deepcopy(padded[-1]) for _ in range(32 - len(padded))]
    adapter = CourseAdapter(padded)
    seed = args.seed + batch
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_output = args.out / f"batch_{batch:03d}"
    batch_output.mkdir()
    env = NativeEnv(
      32,
      seed,
      args.device,
      batch_output,
      evaluation=adapter,
      course=adapter.source_course,
      task=None if args.suite == "general" else args.suite,
    )
    env.policy_identity = manifest["teacher_id"]
    try:
      active = [True] * len(selected)
      horizon = max(row["horizon"] for row in padded)
      limit = (
        min(horizon, args.max_controls) if args.max_controls is not None else horizon
      )
      with torch.no_grad():
        for tick in range(limit):
          episodes = env.frames.episode.cpu().clone()
          valid = env.frames.valid.cpu().clone()
          frame = env.frames.frame.cpu().clone()
          captured = env.frames.capture_tick.cpu().clone()
          action = teacher(env.obs["history"].reshape(-1, 5, 45), env.obs["geometry"])
          _, _, done, info = env.step(action)
          assert adapter.snapshot is not None
          for lane, trial in enumerate(selected):
            if not active[lane]:
              continue
            row = {
              name: value[lane].tolist() for name, value in adapter.snapshot.items()
            }
            row.update(
              tick=tick,
              episode=int(episodes[lane]),
              action=action[lane].tolist(),
              depth_valid=bool(valid[lane]),
              frame=int(frame[lane]),
              capture_tick=int(captured[lane]),
              post_episode=int(env.frames.episode[lane]),
              done=bool(done[lane]),
            )
            trial["rows"].append(row)
            active[lane] = not row["done"]
          if not any(active):
            break
    finally:
      env.close()
  if source_files() != source:
    raise RuntimeError("采集过程中源码改变，结果不可用于资格")
  if teacher_identity(teacher) != manifest["teacher_id"]:
    raise RuntimeError("评估期间教师权重发生变化")
  entries = []
  for trial in trials:
    path = args.out / (trial["spec"]["trial_id"] + ".json")
    write_json(path, trial)
    entries.append(dict(path=path.name, sha256=sha256(path)))
  write_json(
    args.out / "RESULT.json",
    dict(
      manifest_sha256=sha256(args.out / "MANIFEST.json"),
      collection_complete=args.max_controls is None,
      trials=entries,
    ),
  )
  report = score_attempt(args.out / "MANIFEST.json", args.out / "RESULT.json")
  write_json(args.out / "SCORE.json", report)
  return report


def execute(args):
  args.out.mkdir(parents=True, exist_ok=False)
  try:
    return _execute(args)
  except BaseException as error:
    write_json(args.out / "FAILED.json", dict(error=repr(error)))
    raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", choices=SUITES, required=True)
  parser.add_argument(
    "--seed", type=int, choices=EVALUATION_SEEDS, default=EVALUATION_SEEDS[0]
  )
  parser.add_argument("--bundle", type=Path)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--max-controls", type=int)
  parser.add_argument(
    "--max-level", type=int, help="专项0起连续难度前缀；省略为完整家族"
  )
  parser.add_argument("--execute", action="store_true")
  args = parser.parse_args()
  if not args.execute:
    print(
      json.dumps(
        dict(
          executed=False,
          trials=len(roster(args.suite, args.seed, args.max_level)),
          suite=args.suite,
        )
      )
    )
    return
  if (
    args.bundle is None
    or args.out is None
    or (args.max_controls is not None and args.max_controls < 1)
  ):
    parser.error("执行需要bundle和新的out目录；限步必须为正")
  torch.set_num_threads(2)
  with device_lock_path(args.device).open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(json.dumps(execute(args)))


if __name__ == "__main__":
  main()
