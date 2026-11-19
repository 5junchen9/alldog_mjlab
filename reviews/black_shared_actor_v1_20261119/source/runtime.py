"""同一原生环境、同一次相机采样，供教师和学生使用。"""

import sys
from pathlib import Path

import torch
from tensordict import TensorDict
from terrain import geometry, reduce_depth, renderer_frustum

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]


class AlignedFrames:
  """缓存的是输入图像和几何，PPO更新时仍重新计算地形编码器。"""

  def __init__(self, count, device):
    self.tick = 0
    self.episode = torch.zeros(count, dtype=torch.long, device=device)
    self.birth = torch.zeros_like(self.episode)
    self.capture_tick = torch.full_like(self.episode, -1)
    self.frame = torch.full_like(self.episode, -1)
    self.valid = torch.zeros(count, dtype=torch.bool, device=device)
    self.depth = torch.zeros(count, 48, 64, 1, device=device)
    self.geometry = torch.zeros(count, 768, device=device)
    self.sequence = 0

  def due(self):
    due = ((self.tick - self.birth) % 5 == 0) | (self.frame < 0)
    return (due & (self.capture_tick != self.tick)).nonzero().flatten()

  def capture(self, ids, depth, points):
    self.depth[ids] = depth
    self.geometry[ids] = points
    self.capture_tick[ids] = self.tick
    self.frame[ids] = torch.arange(
      self.sequence, self.sequence + len(ids), device=ids.device
    )
    self.sequence += len(ids)
    self.valid[ids] = (torch.isfinite(depth) & (depth > 0)).flatten(1).any(1)

  def advance(self, done):
    self.tick += 1
    ids = done.bool().nonzero().flatten()
    self.episode[ids] += 1
    self.birth[ids] = self.tick
    self.frame[ids] = -1
    self.valid[ids] = False
    self.depth[ids] = 0
    self.geometry[ids] = 0

  def metadata(self):
    return dict(
      tick=self.tick,
      episode=self.episode.clone(),
      capture_episode=self.episode.clone(),
      frame=self.frame.clone(),
      capture_tick=self.capture_tick.clone(),
      mask=self.valid.clone(),
    )


def native_imports():
  # 旧实现作为只读依赖；新入口不通过旧恢复/预算脚本启动。
  directories = [
    BASE / "stairs_passage_course_candidate_20261117",
    BASE
    / "structural_stoprepair_v7_fix1_candidate_20261115/candidate_source/preparation/rough_v2/specialty446_nativecontexts_v34_20261114",
    BASE / "student_direct160_adapter_candidate_20261116",
    BASE / "terrain446_success12335_20261112",
    BASE / "black_reliability_20261119/fault",
    ROOT / "src",
  ]
  sys.path[1:1] = [str(path) for path in directories]


class NativeEnv:
  """保持原奖励、课程、关节动作顺序、缩放和执行器延迟。"""

  evaluation = None

  def __init__(self, count, seed, device, output, evaluation=None):
    native_imports()
    import warp as wp
    from camera_contract import verify_camera
    from chunk_renderer_fixed import ChunkRenderer
    from direct160 import make_renderer
    from parameters import PARAMETERS
    from passage_course import POLICY
    from probe import attach
    from specialty_env import make_specialty_configs, specialty_state

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.velocity.config.black_actuation.actuation import (
      BlackDelayedJointPositionAction,
    )
    from mjlab.tasks.velocity.config.black_teacher.student_env_cfg import (
      black_him_student_env_cfg,
    )

    cfg, self.rl = make_specialty_configs(
      "stairs",
      num_envs=count,
      seed=seed,
      course_parameters=PARAMETERS,
      passage_policy=POLICY,
    )
    cfg.auto_reset = True
    cfg.observations["history"] = black_him_student_env_cfg(
      "Mjlab-Velocity-Flat-Black-Teacher-Actuation"
    ).observations["history"]
    self.evaluation = evaluation
    if evaluation is not None:
      evaluation.configure(cfg)
    previous_spec = cfg.scene.spec_fn

    def add_camera(spec):
      if previous_spec:
        previous_spec(spec)
      spec.body("robot/d435_link").add_camera(
        name="shared_actor_depth",
        pos=[0.10, 0, -0.066],
        quat=[
          0.6163438796401546,
          0.3465836436275126,
          -0.3465836436275126,
          -0.6163438796401546,
        ],
        sensor_size=[424, 240],
        focal_pixel=[223.40138655956403, 216.48573063257084],
        resolution=[424, 240],
      )

    cfg.scene.spec_fn = add_camera
    self.plant = ManagerBasedRlEnv(cfg, device=device)
    self.renderer = self.probe = None
    try:
      self.wrapper = RslRlVecEnvWrapper(self.plant, clip_actions=None)
      self.frames = AlignedFrames(count, device)
      self.count = count
      self.device = device
      self.renderer = None
      self.probe = None
      self.camera_receipt = verify_camera(self.plant.sim.mj_model, "shared_actor_depth")
      term = self.plant.action_manager.get_term("joint_pos")
      if not isinstance(term, BlackDelayedJointPositionAction):
        raise ValueError("需要原Black带执行器延迟的关节位置动作")
      expected = [
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
      ]
      if term.target_names != expected or not bool(
        (torch.as_tensor(term.scale) == 0.25).all()
      ):
        raise ValueError("原生关节顺序或动作缩放改变")
      if not term.cfg.delay_enabled or term.cfg.clear_history_on_reset:
        raise ValueError("执行器延迟约定改变")
      state = specialty_state(self.plant)
      self.metrics = state.episode_metrics
      state.episode_metrics.configure(
        output,
        {
          "policy_kind": "shared_actor_v1_candidate",
          "source_actual_updates": 0,
        },
      )
      cid = self.plant.sim.mj_model.camera("shared_actor_depth").id
      self.renderer, _, self.render_receipt = make_renderer(
        renderer_class=ChunkRenderer,
        mjm=self.plant.sim.mj_model,
        model=self.plant.sim.wp_model,
        data=self.plant.sim.wp_data,
        camera_id=cid,
        chunk=32 if count < 128 else 128,
      )
      self.probe = attach(
        self.plant,
        context_provider=lambda: {
          "tick": self.frames.tick,
          "episode": self.frames.episode,
          "command": self.plant.command_manager.get_command("twist"),
          "policy_kind": "shared_actor_v1_candidate",
          "cold_physics": True,
        },
      )
      self.wp = wp
      self.last_native = self.wrapper.get_observations()
      self.obs = self.bind(self.last_native)
      if evaluation is not None:
        evaluation.attach(self)

    except BaseException as error:
      failure = {"error": repr(error)}
      error.__dict__["failure_details"] = failure
      try:
        if self.probe is not None and self.probe.fault is not None:
          from probe import write_fault

          write_fault(self.probe, Path(output) / "first_fault")
      except BaseException as diagnostic_error:
        failure["diagnostic_error"] = repr(diagnostic_error)
      try:
        self.close()
      except BaseException as close_error:
        failure["close_error"] = repr(close_error)
      raise

  def bind(self, native):
    assert self.renderer is not None
    ids = self.frames.due()
    for selected, raw in self.renderer.frames_for(ids):
      self.wp.synchronize_device(self.plant.sim.wp_device)
      depth = reduce_depth(raw)
      points = geometry(depth, renderer_frustum(self.renderer, selected))
      self.frames.capture(selected, depth, points)
    if (
      not torch.isfinite(native["history"]).all()
      or not torch.isfinite(native["critic"]).all()
    ):
      raise FloatingPointError("原生观测非有限")
    return TensorDict(
      {
        "history": native["history"].flatten(1).clone(),
        "geometry": self.frames.geometry.clone(),
        "critic": native["critic"].clone(),
      },
      batch_size=[self.count],
    )

  def step(self, action):
    if self.evaluation is not None:
      self.evaluation.begin(self.last_native)
    native, reward, done, info = self.wrapper.step(action)
    if "time_outs" in info:
      # 跌倒等真实终止与超时同时发生时，不能给真实终止继续补价值。
      info = dict(info)
      info["time_outs"] = info["time_outs"] & ~self.plant.reset_terminated
    self.last_native = native
    self.frames.advance(done)
    self.obs = self.bind(native)
    if self.evaluation is not None:
      self.evaluation.finish(native, done)
    return self.obs, reward, done, info

  def close(self):
    if self.evaluation is not None:
      self.evaluation.detach()
    if self.probe is not None:
      self.probe.detach()
    if self.renderer is not None:
      self.renderer.contexts.clear()
      self.renderer._staging.clear()
    self.plant.close()
