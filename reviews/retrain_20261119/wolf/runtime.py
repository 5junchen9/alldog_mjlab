"""Wolf原生物理与同帧可见地形；手动reset保留真实终止标签。"""

from pathlib import Path
from types import SimpleNamespace

import torch
from black.runtime import AlignedFrames
from black.runtime import NativeEnv as BlackNativeEnv

from wolf.camera import CAMERA, CAMERA_STATUS, add_camera
from wolf.curriculum import configure
from wolf.fault_capture import attach, write_fault


class NativeEnv(BlackNativeEnv):
  evaluation = None

  def __init__(
    self,
    count,
    seed,
    device,
    output,
    course="flat_foundation",
    evaluation=None,
    max_level=None,
    allow_promotion=True,
  ):
    import warp as wp
    from support.black.chunk_renderer_fixed import ChunkRenderer
    from support.black.direct160 import make_renderer

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    cfg = configure(course, count, seed, max_level, allow_promotion)
    cfg.observations["history"] = cfg.observations.pop("actor")
    if evaluation is not None:
      evaluation.configure(cfg)
    previous_spec = cfg.scene.spec_fn

    def camera_spec(spec):
      if previous_spec is not None:
        previous_spec(spec)
      add_camera(spec)

    cfg.scene.spec_fn = camera_spec
    self.plant = ManagerBasedRlEnv(cfg, device=device)
    self.renderer = self.probe = None
    self.evaluation = evaluation
    self.metrics = SimpleNamespace(policy_update=0)
    self.count, self.device, self.course, self.wp = count, device, course, wp
    try:
      self.wrapper = RslRlVecEnvWrapper(self.plant, clip_actions=None)
      self.frames = AlignedFrames(count, device)
      self.max_episode_controls = int(self.plant.max_episode_length)
      if self.plant.action_manager.total_action_dim != 16:
        raise ValueError("Wolf必须为12腿位置目标+4轮速度目标")
      cid = self.plant.sim.mj_model.camera(CAMERA["name"]).id
      self.camera_receipt = {
        **CAMERA,
        "status": CAMERA_STATUS,
        "control_hz": 50,
        "capture_hz": 10,
      }
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
        lambda: {
          "tick": self.frames.tick,
          "episode": self.frames.episode,
          "command": self.plant.command_manager.get_command("twist"),
          "policy_update": self.metrics.policy_update,
          "policy_identity": self.policy_identity,
        },
      )
      self.last_native = self.wrapper.get_observations()
      self.obs = self.bind(self.last_native)
      if evaluation is not None:
        evaluation.attach(self)
    except BaseException as error:
      details = error.__dict__.setdefault("failure_details", {})
      try:
        if self.probe is not None and self.probe.fault is not None:
          write_fault(self.probe, Path(output) / "first_fault")
      except BaseException as diagnostic_error:
        details["diagnostic_error"] = repr(diagnostic_error)
      try:
        self.close()
      except BaseException as close_error:
        details["close_error"] = repr(close_error)
      raise

  def step(self, action):
    if not bool(torch.isfinite(action).all()):
      raise FloatingPointError("Wolf动作非有限，物理步未执行")
    if self.evaluation is not None:
      self.evaluation.begin(self.last_native)
    native, reward, done, info = self.wrapper.step(action)
    info = dict(info)
    info["terminated"] = self.plant.reset_terminated.clone()
    info["time_outs"] = (self.plant.reset_time_outs & ~info["terminated"]).clone()
    info["terminal_critic"] = native["critic"].clone()
    info["terminal_history"] = native["history"].reshape(self.count, 6, 53).clone()
    # 使用独立estimator_velocity监督组，不直接取critic前三维；目标2×m/s。
    info["next_velocity"] = native["estimator_velocity"].clone()
    reward = reward.clone()
    if not all(
      bool(torch.isfinite(info[name]).all())
      for name in ("terminal_critic", "terminal_history", "next_velocity")
    ):
      raise FloatingPointError("Wolf真实末帧非有限")
    command = self.plant.command_manager.get_term("twist")
    info["route_passed"] = command.passed.clone()
    info["route_failed"] = command.failed.clone()
    info["route_settled"] = command.settled.clone()
    if self.evaluation is not None:
      self.evaluation.capture_terminal()
    ids = done.bool().nonzero().flatten()
    if len(ids):
      self.plant.recorder_manager.record_pre_reset(ids)
      reset_native, _ = self.plant.reset(env_ids=ids)
      for name, values in reset_native.items():
        native[name][ids] = values[ids]
    self.last_native = native
    self.frames.advance(done)
    self.obs = self.bind(native)
    if self.evaluation is not None:
      self.evaluation.finish(native, done)
    return self.obs, reward, done, info
