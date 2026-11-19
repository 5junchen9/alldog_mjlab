"""同一原生环境、同一次相机采样，供教师和学生使用。"""

from pathlib import Path

import torch
from tensordict import TensorDict

from black.terrain import geometry, reduce_depth, renderer_frustum

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "support/black"


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
  """明确包导入不改变sys.path；保留无副作用的测试入口。"""
  return None


class NativeEnv:
  """平地起步；先保留真实末帧，再局部reset，动作与延迟约定明确。"""

  evaluation = None
  policy_identity: str | None = None

  def __init__(
    self,
    count,
    seed,
    device,
    output,
    evaluation=None,
    course="flat_foundation",
    task=None,
    max_level=None,
  ):
    native_imports()
    import warp as wp
    from support.black.camera_contract import verify_camera
    from support.black.chunk_renderer_fixed import ChunkRenderer
    from support.black.config import history_observations, make_general_configs
    from support.black.direct160 import make_renderer
    from support.black.parameters import PARAMETERS
    from support.black.passage_course import POLICY
    from support.black.specialty_env import make_specialty_configs, specialty_state

    from black.curriculum import configure
    from black.fault_capture import attach
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.velocity.config.black_actuation.actuation import (
      BlackDelayedJointPositionAction,
    )

    self.task = task
    self.max_level = max_level
    source_course = (
      getattr(evaluation, "source_course", "specialty")
      if evaluation is not None
      else course
    )
    if source_course == "specialty":
      selected_task = task or "stairs"
      cfg, self.rl = make_specialty_configs(
        selected_task,
        num_envs=count,
        seed=seed,
        course_parameters=PARAMETERS,
        passage_policy=POLICY if selected_task == "stairs" else None,
      )
    else:
      cfg, self.rl = make_general_configs(count, seed)
    # 手动局部reset让HIM标签和timeout价值读取同一真实末帧。
    cfg.auto_reset = False
    cfg.observations["history"] = history_observations()
    self.evaluation = evaluation
    self.course = course
    configure(cfg, source_course)
    if max_level is not None:
      from black.curriculum import limit_student_specialty

      if source_course != "specialty" or evaluation is not None:
        raise ValueError("难度上限只用于已准入的专项学生")
      limit_student_specialty(cfg, max_level)
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
      self.max_episode_controls = int(self.plant.max_episode_length)
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
      if hasattr(cfg.commands["twist"], "task"):
        self.metrics = specialty_state(self.plant).episode_metrics
      else:
        from support.black.stair_metrics import EpisodeMetrics

        self.metrics = EpisodeMetrics(self.plant)
      self.metrics.configure(
        output,
        {
          "policy_kind": "shared_actor_v2_scratch",
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
          "policy_kind": "shared_actor_v2_scratch",
          "cold_physics": True,
          "policy_identity": getattr(self, "policy_identity", None),
          "policy_update": self.metrics.policy_update,
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
          from black.fault_capture import write_fault

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
      depth, source_pixels = reduce_depth(raw)
      points = geometry(depth, source_pixels, renderer_frustum(self.renderer, selected))
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
    if not bool(torch.isfinite(action).all()):
      bad = (~torch.isfinite(action).all(-1)).nonzero().flatten()
      error = FloatingPointError("动作含NaN/Inf，物理步未执行")
      error.__dict__["failure_details"] = {
        "stage": "policy_action_before_physics",
        "bad_environment_ids": bad.cpu().tolist(),
        "requested_actions_first_8": repr(action[bad[:8]].detach().cpu().tolist()),
      }
      raise error
    if self.evaluation is not None:
      self.evaluation.begin(self.last_native)
    native, reward, done, info = self.wrapper.step(action)
    # reset会改写原生buffer；先复制终止标志、奖励和监督目标。
    info = dict(info)
    info["terminated"] = self.plant.reset_terminated.clone()
    info["time_outs"] = (self.plant.reset_time_outs & ~info["terminated"]).clone()
    info["terminal_critic"] = native["critic"].clone()
    info["terminal_history"] = native["history"].reshape(self.count, 5, 45).clone()
    info["next_velocity"] = info["terminal_critic"][:, :3].clone()
    reward = reward.clone()
    if not all(
      bool(torch.isfinite(info[name]).all())
      for name in ("terminal_critic", "terminal_history", "next_velocity")
    ):
      raise FloatingPointError("真实末帧非有限，停止并保留原始故障")
    ids = done.bool().nonzero().flatten()
    if len(ids):
      self.plant.recorder_manager.record_pre_reset(ids)
      reset_native, _ = self.plant.reset(env_ids=ids)
      # 非reset lane使用本次真实step的观测，避免局部reset重算噪声污染它们。
      for name, values in reset_native.items():
        assert isinstance(values, torch.Tensor)
        native[name][ids] = values[ids]
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
