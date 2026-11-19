"""学生实际控制采样；冻结教师在相同访问状态上提供监督。"""

from pathlib import Path

import torch

from black.checkpoint import save_bundle, teacher_identity
from black.learner_state import (
  SCHEMA,
  capture_rng,
  parameter_signature,
  restore_rng,
  validate_resume,
)
from black.student import SharedStudent, distillation_loss

# isort: split
# GRU状态由教师和学生共同使用的独立支持模块定义。
from support.black.causal_visual_gru import State

CONTROL_STEPS = 24
LANES_PER_BATCH = 128
LEARNING_RATE = 1e-4
CAMERA_INTERVAL_CONTROLS = 5


def _prefix_memory(count, max_controls, history_width=225, geometry_width=768):
  """按有限episode与10Hz相机估算CPU峰值，开采样前拦截明显内存不足。"""
  max_frames = (max_controls + CAMERA_INTERVAL_CONTROLS - 1) // CAMERA_INTERVAL_CONTROLS
  # 每帧2048字节为Python元数据/容器余量；采样stack时会短暂保留两份本轮数据。
  prefix_bytes = count * max_frames * (48 * 64 * 4 + 2048)
  round_bytes = count * CONTROL_STEPS * (history_width + geometry_width + 48 * 64) * 4
  estimate = prefix_bytes + 2 * round_bytes
  memory = Path("/proc/meminfo").read_text().splitlines()
  available = next(
    int(line.split()[1]) * 1024 for line in memory if line.startswith("MemAvailable:")
  )
  if estimate > available * 0.8:
    raise MemoryError(
      f"full episode prefix needs approximately {estimate} CPU bytes; "
      f"available {available}; reduce num_envs before starting"
    )
  return max_frames, prefix_bytes, estimate


def _state_slice(state, start, stop, device):
  if state is None:
    return None
  return State(
    episode=state.episode[start:stop].detach().to(device),
    frame=state.frame[start:stop].detach().to(device),
    capture_tick=state.capture_tick[start:stop].detach().to(device),
    tick=state.tick,
    hidden=state.hidden[start:stop].detach().to(device),
  )


def _metadata_slice(metadata, start, stop, device):
  return {
    name: value[start:stop].to(device) if torch.is_tensor(value) else value
    for name, value in metadata.items()
  }


def _collect(env, student, state):
  history, geometry, depth, metadata = [], [], [], []
  student.eval()
  with torch.no_grad():
    for _ in range(CONTROL_STEPS):
      current = env.obs["history"].reshape(env.count, *student.teacher.history_shape)
      row = env.frames.metadata()
      # 在step之前复制，防止环境复用观测/帧缓冲区改变旧样本。
      history.append(current.detach().cpu().clone())
      geometry.append(env.obs["geometry"].detach().cpu().clone())
      depth.append(env.frames.depth.detach().cpu().clone())
      metadata.append(
        {
          name: value.detach().cpu().clone() if torch.is_tensor(value) else value
          for name, value in row.items()
        }
      )
      action, _, state = student(current, env.frames.depth, row, state)
      env.step(action)
  return {
    "history": torch.stack(history),
    "geometry": torch.stack(geometry),
    "depth": torch.stack(depth),
    "metadata": metadata,
  }, state


def _sequence(student, sample, prefix, start, stop, device):
  history = sample["history"][:, start:stop].to(device)
  depth = sample["depth"][:, start:stop].to(device)
  metadata = [_metadata_slice(row, start, stop, device) for row in sample["metadata"]]
  state = _state_slice(prefix, start, stop, device)
  actions, visual, state = student.forward_sequence(history, depth, metadata, state)
  del actions  # 监督使用相同visual重新过冻结actor，保留这条梯度路径。
  return history, visual, state


class EpisodePrefix:
  """CPU保存每lane本episode所有fresh valid帧；不截断、无旧参数hidden。

  GRU在held/invalid步不更新hidden，因此完整重算只需实际更新过的相机帧。
  控制时钟和episode身份仍逐步检查；到episode时长上限未reset就停止。
  """

  def __init__(self, count, max_controls, max_age):
    if max_controls < 1:
      raise ValueError("a finite positive episode control limit is required")
    self.max_controls = max_controls
    self.max_age = max_age
    self.max_frames = (
      max_controls + CAMERA_INTERVAL_CONTROLS - 1
    ) // CAMERA_INTERVAL_CONTROLS
    self.rows = [[] for _ in range(count)]
    self.episode = [-1] * count
    self.birth = [0] * count
    self.last_frame = [-1] * count

  def extend(self, sample):
    for depth, metadata in zip(sample["depth"], sample["metadata"], strict=True):
      tick = metadata["tick"]
      for lane in range(len(self.rows)):
        episode = int(metadata["episode"][lane])
        if episode != self.episode[lane]:
          self.rows[lane].clear()
          self.episode[lane] = episode
          self.birth[lane] = tick
          self.last_frame[lane] = -1
        if tick - self.birth[lane] >= self.max_controls:
          raise RuntimeError("episode exceeded declared prefix replay memory bound")
        frame = int(metadata["frame"][lane])
        age = tick - int(metadata["capture_tick"][lane])
        valid = (
          bool(metadata["mask"][lane])
          and int(metadata["capture_episode"][lane]) == episode
          and frame >= 0
          and 0 <= age <= self.max_age
        )
        if valid and frame != self.last_frame[lane]:
          if self.rows[lane]:
            previous_capture = self.rows[lane][-1][1]["capture_tick"]
            if (
              int(metadata["capture_tick"][lane]) - previous_capture
              < CAMERA_INTERVAL_CONTROLS
            ):
              raise RuntimeError("camera cadence exceeds declared prefix memory bound")
          if len(self.rows[lane]) >= self.max_frames:
            raise RuntimeError("fresh frame count exceeded episode prefix bound")
          row = {
            name: int(value[lane]) if torch.is_tensor(value) else value
            for name, value in metadata.items()
          }
          # clone解除与整轮depth tensor的引用，reset即可释放本lane旧episode。
          self.rows[lane].append((depth[lane].clone(), row))
          self.last_frame[lane] = frame

  def refresh(self, student, state, device):
    """当前参数从零重算每个episode全部fresh帧；不回放物理或监督标签。"""
    student.eval()
    parts = []
    with torch.no_grad():
      for start in range(0, len(self.rows), LANES_PER_BATCH):
        rows = self.rows[start : start + LANES_PER_BATCH]
        hidden = torch.zeros(len(rows), 64, device=device)
        for number in range(max(map(len, rows))):
          lanes = [lane for lane, events in enumerate(rows) if number < len(events)]
          events = [rows[lane][number] for lane in lanes]
          depth = torch.stack([event[0] for event in events]).to(device)
          metadata = {
            name: torch.tensor([event[1][name] for event in events], device=device)
            for name in events[0][1]
          }
          metadata["mask"] = metadata["mask"].bool()
          encoded = student.encode_depth(depth, metadata)
          ids = torch.tensor(lanes, device=device)
          hidden = hidden.index_copy(0, ids, student.gru.gru(encoded, hidden[ids]))
        parts.append(hidden)
    return State(
      episode=state.episode,
      frame=state.frame,
      capture_tick=state.capture_tick,
      tick=state.tick,
      hidden=torch.cat(parts),
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
):
  """外部检查教师资格后调用；不在这里创建或推断能力验收结果。

  每轮24步、每128lane做一次有限序列BPTT，累积后仅执行一次Adam。
  BPTT梯度仍截断在本轮起点；更新后从episode起点重算全部视觉hidden。
  重算不使用旧参数前缀，所需CPU缓存受原生最大episode时长约束。
  rounds是本次新增轮数。resume仅恢复学习器；调用方先加载该包的模型，
  物理、课程、相机和GRU记忆均重新初始化，不续接上次物理轨迹。
  """
  if rounds < 1:
    raise ValueError("at least one student round required")
  if save_every < 1:
    raise ValueError("save_every must be >=1")
  max_controls = getattr(env, "max_episode_controls", None)
  if max_controls is None:
    max_controls = env.plant.max_episode_length
  max_controls = int(max_controls)
  max_frames, prefix_bytes, memory_estimate = _prefix_memory(env.count, max_controls)
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  student = SharedStudent(teacher, cnn, gru)
  env.policy_identity = teacher_identity(teacher)
  # Adam状态按参数位置绑定，因此记录名称并保持与实际optimizer完全相同的顺序。
  named_parameters = [
    (f"depth_encoder.{name}", parameter)
    for name, parameter in student.depth_encoder.named_parameters()
  ] + [(f"gru.{name}", parameter) for name, parameter in student.gru.named_parameters()]
  parameters = [parameter for _, parameter in named_parameters]
  signature = parameter_signature(named_parameters)
  gru_cell_parameters = set(student.gru.gru.parameters())
  optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
  recipe = {
    "controls_per_round": CONTROL_STEPS,
    "lanes_per_microbatch": LANES_PER_BATCH,
    "num_envs": env.count,
    "course": env.course,
    "task": getattr(env, "task", None),
    "max_level": getattr(env, "max_level", None),
    "optimizer": "Adam",
    "learning_rate": LEARNING_RATE,
    "gradient_clip_norm": 1.0,
    "optimizer_updates_per_round": 1,
    "feature_weight": float(feature_weight),
    "loss": "Huber(action_mean)+feature_weight*Huber(terrain32)",
    "recipe_status": "initial_experiment_not_tuned",
    "controller": "student",
    "teacher_queries": "same_student_visited_state",
    "trainable": ["CNN", "GRU"],
    "recurrent_context": "current_parameters_replay_all_fresh_frames_since_episode_start",
    "gradient_context": "BPTT_truncated_at_round_start",
    "exact_long_episode_replay": True,
    "prefix_max_episode_controls": max_controls,
    "prefix_max_fresh_frames_per_lane": max_frames,
    "prefix_cpu_bytes_estimate": prefix_bytes,
    "sampling_and_prefix_cpu_bytes_estimate": memory_estimate,
    "prefix_memory_admission": "estimate_must_not_exceed_80_percent_MemAvailable",
    "prefix_storage": "CPU_FP32_fresh_valid_frames_and_original_metadata",
  }
  completed_updates, optimizer_calls = 0, 0
  if resume is not None:
    if teacher_identity(teacher) != resume.get("teacher_id"):
      raise ValueError("student resume teacher identity mismatch")
    from black.checkpoint import source_identity

    learning_state = validate_resume(
      resume, "student", recipe, current_source=source_identity()
    )
    if learning_state.get("parameter_signature") != signature:
      raise ValueError("student optimizer parameter signature mismatch")
    completed_updates = resume["completed_updates"]
    optimizer_calls = learning_state["optimizer_calls"]
    if optimizer_calls != completed_updates:
      raise ValueError("student requires one optimizer call per completed update")
    optimizer.load_state_dict(learning_state["optimizer"])
    restore_rng(learning_state["rng"], env.device)

  def save(path):
    return save_bundle(
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
        "restart_boundary": "new_physics_curriculum_camera_gru_episode",
      },
    )

  # 不加载上一进程的hidden；旧帧身份和新物理episode不能拼接。
  state = None
  episode_prefix = EpisodePrefix(env.count, max_controls, student.gru.max_age)
  starting_updates = completed_updates
  metrics = []
  periodic_checkpoints = []
  for _ in range(rounds):
    env.metrics.policy_update = completed_updates
    prefix = _state_slice(state, 0, env.count, "cpu")
    sample, state = _collect(env, student, state)
    episode_prefix.extend(sample)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    summary = {
      "round": completed_updates + 1,
      "loss": 0.0,
      "action_huber": 0.0,
      "feature_huber": 0.0,
    }
    gru_cell_used = False

    def mark_gru_cell_used(_module, _inputs):
      nonlocal gru_cell_used
      gru_cell_used = True

    # 没有fresh valid帧时，原GRU按设计不调用GRUCell。
    # 只记录本轮反向重放是否实际调用它，异常退出也会移除hook。
    with student.gru.gru.register_forward_pre_hook(mark_gru_cell_used):
      for start in range(0, env.count, LANES_PER_BATCH):
        stop = min(start + LANES_PER_BATCH, env.count)
        history, visual, _ = _sequence(student, sample, prefix, start, stop, env.device)
        geometry = sample["geometry"][:, start:stop].to(env.device)
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
        # 每项loss已对24步和该微批取均值，最后不足128lane也按实际数量加权。
        weight = (stop - start) / env.count
        (loss * weight).backward()
        summary["loss"] += float(loss.detach()) * weight
        for name in ("action_huber", "feature_huber"):
          summary[name] += float(values[name]) * weight
    if any(
      parameter.grad is None and (gru_cell_used or parameter not in gru_cell_parameters)
      for parameter in parameters
    ):
      raise RuntimeError("missing student gradient")
    # 保留未使用GRUCell的None梯度，Adam会跳过这些参数及其旧动量。
    summary["gru_cell_used"] = gru_cell_used
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer_calls += 1
    completed_updates += 1
    metrics.append(summary)
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
    "cold_start": ["physics", "curriculum", "camera", "GRU"],
    "exact_physical_resume": False,
  }
