"""学生实际控制采样；冻结教师在相同访问状态上提供监督。"""

import json
from pathlib import Path

import torch
from support.black.causal_visual_gru import State

from black.learner_state import (
  SCHEMA,
  capture_rng,
  parameter_signature,
  restore_rng,
  validate_resume,
)
from black.student import SharedStudent, distillation_loss

CONTROL_STEPS = 24
LANES_PER_BATCH = 256
REFRESH_BATCH = 512
LEARNING_RATE = 3e-4
DEFAULT_EPOCHS = 4
DEFAULT_PREFIX_WINDOW = 50
CAMERA_INTERVAL_CONTROLS = 5


def _prefix_memory(
  count,
  max_controls,
  max_prefix_frames=DEFAULT_PREFIX_WINDOW,
  history_width=225,
  geometry_width=768,
):
  """按声明的视觉记忆窗口与10Hz相机估算CPU峰值，开采样前拦截内存不足。"""
  stored_frames = int(max_prefix_frames)
  # 向量化连续张量存储：48x64 FP32深度 + 6个int64/bool元数据槽位，无Python dict开销。
  prefix_bytes = count * stored_frames * (48 * 64 * 4 + 64)
  round_bytes = count * CONTROL_STEPS * (history_width + geometry_width + 48 * 64) * 4
  estimate = prefix_bytes + 2 * round_bytes
  memory = Path("/proc/meminfo").read_text().splitlines()
  available = next(
    int(line.split()[1]) * 1024 for line in memory if line.startswith("MemAvailable:")
  )
  if estimate > available * 0.8:
    raise MemoryError(
      f"visual prefix buffer needs approximately {estimate} CPU bytes; "
      f"available {available}; reduce num_envs or prefix window before starting"
    )
  return stored_frames, prefix_bytes, estimate


def _state_slice(state, indices, device):
  if state is None:
    return None
  return State(
    episode=state.episode[indices].detach().to(device),
    frame=state.frame[indices].detach().to(device),
    capture_tick=state.capture_tick[indices].detach().to(device),
    tick=state.tick,
    hidden=state.hidden[indices].detach().to(device),
    fresh_count=state.fresh_count[indices].detach().to(device),
    warm_hidden=state.warm_hidden[indices].detach().to(device),
  )


def _metadata_slice(metadata, indices, device):
  return {
    name: value[indices].to(device) if torch.is_tensor(value) else value
    for name, value in metadata.items()
  }


def _collect(env, student, state):
  history, geometry, depth, metadata = [], [], [], []
  student.eval()
  with torch.no_grad():
    for _ in range(CONTROL_STEPS):
      current = env.obs["history"].reshape(env.count, *student.teacher.history_shape)
      row = env.frames.metadata()
      frame_depth = env.frames.depth
      # 在step之前复制，防止环境复用观测/帧缓冲区改变旧样本；采集与监督共用同帧深度。
      history.append(current.detach().cpu().clone())
      geometry.append(env.obs["geometry"].detach().cpu().clone())
      depth.append(frame_depth.detach().cpu().clone())
      metadata.append(
        {
          name: value.detach().cpu().clone() if torch.is_tensor(value) else value
          for name, value in row.items()
        }
      )
      action, _, state = student(current, frame_depth, row, state)
      env.step(action)
  return {
    "history": torch.stack(history),
    "geometry": torch.stack(geometry),
    "depth": torch.stack(depth),
    "metadata": metadata,
  }, state


def _sequence(student, sample, prefix, indices, device):
  history = sample["history"][:, indices].to(device)
  depth = sample["depth"][:, indices].to(device)
  metadata = [_metadata_slice(row, indices, device) for row in sample["metadata"]]
  state = _state_slice(prefix, indices, device)
  actions, visual, state = student.forward_sequence(history, depth, metadata, state)
  del actions  # 监督使用相同visual重新过冻结actor，保留这条梯度路径。
  return history, visual, state


class EpisodePrefix:
  """保存当前重叠记忆段；满50帧保留后25帧，再追加下一帧。"""

  def __init__(
    self, count, max_controls, max_age, max_prefix_frames=DEFAULT_PREFIX_WINDOW
  ):
    if max_controls < 1:
      raise ValueError("a finite positive episode control limit is required")
    if max_prefix_frames < 2 or max_prefix_frames % 2:
      raise ValueError("max_prefix_frames must be an even window >=2")
    self.count = count
    self.max_controls = max_controls
    self.max_age = max_age
    self.capacity = int(max_prefix_frames)
    self.depth_buf = torch.zeros(count, self.capacity, 48, 64, 1, dtype=torch.float32)
    self.tick_buf = torch.zeros(count, self.capacity, dtype=torch.long)
    self.episode_buf = torch.zeros(count, self.capacity, dtype=torch.long)
    self.capture_episode_buf = torch.zeros(count, self.capacity, dtype=torch.long)
    self.frame_buf = torch.full((count, self.capacity), -1, dtype=torch.long)
    self.capture_tick_buf = torch.full((count, self.capacity), -1, dtype=torch.long)
    self.mask_buf = torch.zeros(count, self.capacity, dtype=torch.bool)
    self.length = torch.zeros(count, dtype=torch.long)
    self.episode = torch.full((count,), -1, dtype=torch.long)
    self.birth = torch.zeros(count, dtype=torch.long)
    self.last_frame = torch.full((count,), -1, dtype=torch.long)
    self.last_capture_tick = torch.full((count,), -1, dtype=torch.long)

  def lane_ticks(self, lane: int) -> list[int]:
    n = int(self.length[lane])
    return self.tick_buf[lane, :n].tolist()

  def extend(self, sample):
    for depth, metadata in zip(sample["depth"], sample["metadata"], strict=True):
      tick = int(metadata["tick"])
      ep = metadata["episode"].to(dtype=torch.long, device="cpu")
      cap_ep = metadata["capture_episode"].to(dtype=torch.long, device="cpu")
      frame = metadata["frame"].to(dtype=torch.long, device="cpu")
      cap_tick = metadata["capture_tick"].to(dtype=torch.long, device="cpu")
      mask = metadata["mask"].to(dtype=torch.bool, device="cpu")
      changed = ep != self.episode
      if bool(changed.any()):
        self.length[changed] = 0
        self.episode[changed] = ep[changed]
        self.birth[changed] = tick
        self.last_frame[changed] = -1
        self.last_capture_tick[changed] = -1
      if bool(((tick - self.birth) >= self.max_controls).any()):
        raise RuntimeError("episode exceeded declared prefix replay memory bound")
      age = tick - cap_tick
      valid = mask & (cap_ep == ep) & (frame >= 0) & (age >= 0) & (age <= self.max_age)
      fresh = valid & (frame != self.last_frame)
      if not bool(fresh.any()):
        continue
      has_prev = fresh & (self.length > 0)
      if bool(has_prev.any()) and bool(
        (
          cap_tick[has_prev] - self.last_capture_tick[has_prev]
          < CAMERA_INTERVAL_CONTROLS
        ).any()
      ):
        raise RuntimeError("camera cadence exceeds declared prefix memory bound")
      lanes = fresh.nonzero().flatten()
      full = self.length[lanes] >= self.capacity
      retained = self.capacity // 2
      full_lanes = lanes[full]
      for buffer in (
        self.depth_buf,
        self.tick_buf,
        self.episode_buf,
        self.capture_episode_buf,
        self.frame_buf,
        self.capture_tick_buf,
        self.mask_buf,
      ):
        buffer[full_lanes, :retained] = buffer[full_lanes, retained:]
      self.length[full_lanes] = retained
      slot = self.length[lanes]
      self.length[lanes] += 1
      self.depth_buf[lanes, slot] = depth[lanes]
      self.tick_buf[lanes, slot] = tick
      self.episode_buf[lanes, slot] = ep[lanes]
      self.capture_episode_buf[lanes, slot] = cap_ep[lanes]
      self.frame_buf[lanes, slot] = frame[lanes]
      self.capture_tick_buf[lanes, slot] = cap_tick[lanes]
      self.mask_buf[lanes, slot] = True
      self.last_frame[lanes] = frame[lanes]
      self.last_capture_tick[lanes] = cap_tick[lanes]

  def refresh(self, student, state, device, indices=None):
    """当前参数从零重算旧段和预热段；无梯度burn-in，最多50帧。"""
    student.eval()
    parts, warm_parts = [], []
    indices = torch.arange(self.count) if indices is None else indices
    with torch.no_grad():
      for start in range(0, len(indices), REFRESH_BATCH):
        lanes = indices[start : start + REFRESH_BATCH]
        lengths = self.length[lanes]
        hidden = torch.zeros(len(lanes), 64, device=device)
        warm_hidden = torch.zeros_like(hidden)
        max_len = int(lengths.max()) if len(lengths) else 0
        for step in range(max_len):
          local_ids = (lengths > step).nonzero().flatten()
          global_ids = lanes[local_ids]
          slots = torch.full_like(global_ids, step)
          depth = self.depth_buf[global_ids, slots].to(device)
          metadata = {
            "tick": self.tick_buf[global_ids, slots].to(device),
            "episode": self.episode_buf[global_ids, slots].to(device),
            "capture_episode": self.capture_episode_buf[global_ids, slots].to(device),
            "frame": self.frame_buf[global_ids, slots].to(device),
            "capture_tick": self.capture_tick_buf[global_ids, slots].to(device),
            "mask": self.mask_buf[global_ids, slots].to(device),
          }
          encoded = student.encode_depth(depth, metadata)
          ids_dev = local_ids.to(device)
          hidden = hidden.index_copy(
            0, ids_dev, student.gru.gru(encoded, hidden[ids_dev])
          )
          if step >= self.capacity // 2:
            warm_hidden = warm_hidden.index_copy(
              0, ids_dev, student.gru.gru(encoded, warm_hidden[ids_dev])
            )
        parts.append(hidden)
        warm_parts.append(warm_hidden)
    return State(
      episode=state.episode,
      frame=state.frame,
      capture_tick=state.capture_tick,
      tick=state.tick,
      hidden=torch.cat(parts),
      fresh_count=state.fresh_count,
      warm_hidden=torch.cat(warm_parts),
    )


def train_student(
  env,
  teacher,
  cnn,
  gru,
  rounds,
  output,
  feature_weight=1.0,
  *,
  resume: dict | None = None,
  save_every: int = 100,
  learning_rate: float = LEARNING_RATE,
  epochs: int = DEFAULT_EPOCHS,
  minibatch_lanes: int = LANES_PER_BATCH,
  max_prefix_frames: int = DEFAULT_PREFIX_WINDOW,
  checkpoint_api=None,
):
  """外部检查教师资格后调用；不在这里创建或推断能力验收结果。

  每轮24步采集后按minibatch进行多epoch有限序列BPTT更新，更新后用向量化环缓存
  以当前参数重算重叠GRU记忆；采集与部署都在满窗后提升已预热的后半段。
  """
  if rounds < 1:
    raise ValueError("at least one student round required")
  if save_every < 1 or epochs < 1 or minibatch_lanes < 1:
    raise ValueError("save_every, epochs and minibatch_lanes must be >=1")
  if max_prefix_frames != gru.memory_frames:
    raise ValueError("prefix容量必须与推理GRU的记忆段长度相同")
  if checkpoint_api is None:
    from black import checkpoint as checkpoint_api
  max_controls = getattr(env, "max_episode_controls", None)
  if max_controls is None:
    max_controls = env.plant.max_episode_length
  max_controls = int(max_controls)
  history_shape = getattr(teacher, "history_shape", None)
  history_width = (
    225 if history_shape is None else int(history_shape[0] * history_shape[1])
  )
  max_frames, prefix_bytes, memory_estimate = _prefix_memory(
    env.count,
    max_controls,
    max_prefix_frames=max_prefix_frames,
    history_width=history_width,
  )
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  student = SharedStudent(teacher, cnn, gru)
  env.policy_identity = checkpoint_api.teacher_identity(teacher)
  initial_teacher_id = env.policy_identity
  # Adam状态按参数位置绑定，因此记录名称并保持与实际optimizer完全相同的顺序。
  named_parameters = [
    (f"depth_encoder.{name}", parameter)
    for name, parameter in student.depth_encoder.named_parameters()
  ] + [(f"gru.{name}", parameter) for name, parameter in student.gru.named_parameters()]
  parameters = [parameter for _, parameter in named_parameters]
  signature = parameter_signature(named_parameters)
  gru_cell_parameters = set(student.gru.gru.parameters())
  optimizer = torch.optim.Adam(parameters, lr=learning_rate)
  minibatches_per_epoch = (env.count + minibatch_lanes - 1) // minibatch_lanes
  recipe = {
    "controls_per_round": CONTROL_STEPS,
    "lanes_per_microbatch": int(minibatch_lanes),
    "epochs_per_round": int(epochs),
    "num_envs": env.count,
    "course": env.course,
    "task": getattr(env, "task", None),
    "max_level": getattr(env, "max_level", None),
    "optimizer": "Adam",
    "learning_rate": float(learning_rate),
    "gradient_clip_norm": 1.0,
    "optimizer_updates_per_round": int(epochs * minibatches_per_epoch),
    "feature_weight": float(feature_weight),
    "camera_noise": getattr(env, "training_sensors", False),
    "loss": "Huber(action_mean)+feature_weight*Huber(terrain32)",
    "controller": "student",
    "teacher_queries": "same_student_visited_state",
    "trainable": ["CNN", "GRU"],
    "recurrent_context": "current_parameters_replay_50_frame_segment_with_25_frame_overlap",
    "gradient_context": "BPTT_truncated_at_round_start",
    "exact_long_episode_replay": False,
    "memory_boundary": "half_window_warmup_crossfade_then_handoff; zero_only_on_episode_reset",
    "prefix_window_frames": max_prefix_frames,
    "prefix_max_episode_controls": max_controls,
    "prefix_max_fresh_frames_per_lane": max_frames,
    "prefix_cpu_bytes_estimate": prefix_bytes,
    "sampling_and_prefix_cpu_bytes_estimate": memory_estimate,
    "prefix_memory_admission": "estimate_must_not_exceed_80_percent_MemAvailable",
    "prefix_storage": "CPU_FP32_vectorized_current_segment",
  }
  completed_updates, optimizer_calls = 0, 0
  if resume is not None:
    if checkpoint_api.teacher_identity(teacher) != resume.get("teacher_id"):
      raise ValueError("student resume teacher identity mismatch")
    learning_state = validate_resume(
      resume, "student", recipe, current_source=checkpoint_api.source_identity()
    )
    if learning_state.get("parameter_signature") != signature:
      raise ValueError("student optimizer parameter signature mismatch")
    completed_updates = resume["completed_updates"]
    optimizer_calls = learning_state["optimizer_calls"]
    expected_calls = completed_updates * recipe["optimizer_updates_per_round"]
    if optimizer_calls != expected_calls:
      raise ValueError("student optimizer call count disagrees with completed updates")
    optimizer.load_state_dict(learning_state["optimizer"])
    if hasattr(env, "restore_curriculum"):
      env.restore_curriculum(learning_state["curriculum"])
    restore_rng(learning_state["rng"], env.device)

  def save(path):
    if checkpoint_api.teacher_identity(teacher) != initial_teacher_id:
      raise RuntimeError("学生更新改变了冻结教师")
    return checkpoint_api.save_bundle(
      path,
      teacher,
      cnn,
      gru,
      phase="student",
      updates=completed_updates,
      learning_state={
        "schema": SCHEMA,
        "recipe": recipe,
        "optimizer": optimizer.state_dict(),
        "optimizer_calls": optimizer_calls,
        "rng": capture_rng(env.device),
        "parameter_signature": signature,
        "curriculum": env.capture_curriculum()
        if hasattr(env, "capture_curriculum")
        else None,
        "restart_boundary": "new_physics_camera_gru_episode_preserved_curriculum",
      },
    )

  # 不加载上一进程的hidden；旧帧身份和新物理episode不能拼接。
  state = None
  episode_prefix = EpisodePrefix(
    env.count, max_controls, student.gru.max_age, max_prefix_frames=max_prefix_frames
  )
  starting_updates = completed_updates
  metrics = []
  periodic_checkpoints = []
  for _ in range(rounds):
    env.metrics.policy_update = completed_updates
    prefix = _state_slice(state, slice(0, env.count), "cpu")
    sample, state = _collect(env, student, state)
    student.train()
    summary = {
      "round": completed_updates + 1,
      "loss": 0.0,
      "action_huber": 0.0,
      "feature_huber": 0.0,
    }
    round_gru_cell_used = False
    for _epoch in range(epochs):
      order = (
        torch.randperm(env.count)
        if epochs > 1 or minibatch_lanes < env.count
        else torch.arange(env.count)
      )
      for start in range(0, env.count, minibatch_lanes):
        stop = min(start + minibatch_lanes, env.count)
        batch_ids = order[start:stop]
        # 每次Adam前以当前参数重算burn-in；不得复用采样时的旧参数hidden。
        batch_prefix = _state_slice(prefix, batch_ids, env.device)
        if batch_prefix is not None:
          batch_prefix = episode_prefix.refresh(
            student, batch_prefix, env.device, batch_ids
          )
        student.train()
        optimizer.zero_grad(set_to_none=True)
        gru_cell_used = False

        def mark_gru_cell_used(_module, _inputs):
          nonlocal gru_cell_used
          gru_cell_used = True

        with student.gru.gru.register_forward_pre_hook(mark_gru_cell_used):
          history = sample["history"][:, batch_ids].to(env.device)
          depths = sample["depth"][:, batch_ids].to(env.device)
          rows = [
            _metadata_slice(row, batch_ids, env.device) for row in sample["metadata"]
          ]
          _, visual, _ = student.forward_sequence(history, depths, rows, batch_prefix)
          geometry = sample["geometry"][:, batch_ids].to(env.device)
          loss, values = distillation_loss(
            history.flatten(0, 1),
            geometry.flatten(0, 1),
            visual.flatten(0, 1),
            student,
            teacher,
            feature_weight,
          )
          if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite student loss")
          loss.backward()
        if any(
          parameter.grad is None
          and (gru_cell_used or parameter not in gru_cell_parameters)
          for parameter in parameters
        ):
          raise RuntimeError("missing student gradient")
        # 保留未使用GRUCell的None梯度，Adam会跳过这些参数及其旧动量。
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer_calls += 1
        round_gru_cell_used |= gru_cell_used
        weight = (stop - start) / (env.count * epochs)
        summary["loss"] += float(loss.detach()) * weight
        for name in ("action_huber", "feature_huber"):
          summary[name] += float(values[name]) * weight
    summary["gru_cell_used"] = round_gru_cell_used
    completed_updates += 1
    metrics.append(summary)
    with (output / "updates.jsonl").open("a") as stream:
      stream.write(json.dumps({**summary, "optimizer_calls": optimizer_calls}) + "\n")
    episode_prefix.extend(sample)
    state = episode_prefix.refresh(student, state, env.device)
    if completed_updates % save_every == 0:
      periodic_path = output / f"student_update_{completed_updates:06d}.pt"
      save(periodic_path)
      periodic_checkpoints.append(str(periodic_path))

  path = output / "student.pt"
  assert state is not None  # rounds>=1，每轮采样后都产生对应的GRU状态。
  teacher_id = save(path)
  return {
    "checkpoint": str(path),
    "teacher_id": teacher_id,
    "completed_updates": completed_updates,
    "starting_updates": starting_updates,
    "added_updates": rounds,
    "optimizer_calls": optimizer_calls,
    "periodic_checkpoints": periodic_checkpoints,
    "control_steps": rounds * CONTROL_STEPS,
    "environment_transitions": rounds * CONTROL_STEPS * env.count,
    "final_state_tick": state.tick,
    "recipe": recipe,
    "round_metrics": metrics,
    "qualification": "not_assessed_here",
    "learner_resumed": resume is not None,
    "cold_start": ["physics", "camera", "GRU"],
    "curriculum_restored": resume is not None and hasattr(env, "restore_curriculum"),
    "exact_physical_resume": False,
  }
