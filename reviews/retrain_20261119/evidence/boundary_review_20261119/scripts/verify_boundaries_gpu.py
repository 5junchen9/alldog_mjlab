"""真实GPU空视野与合成连续记忆流；不授予任务资格、不保存训练权重。"""

import argparse
import fcntl
import importlib
import json
import sys
from pathlib import Path

import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from black.checkpoint import tensor_digest  # noqa: E402
from black.eval_collect import device_lock_path  # noqa: E402
from black.runtime import AlignedFrames  # noqa: E402
from black.student import SharedStudent, distillation_loss  # noqa: E402
from black.student_training import EpisodePrefix  # noqa: E402


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("robot", choices=("black", "wolf"))
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)
  checkpoint = importlib.import_module(args.robot + ".checkpoint")
  runtime = importlib.import_module(args.robot + ".runtime")
  torch.manual_seed(191119)
  torch.set_num_threads(2)
  source = checkpoint.source_identity()
  with device_lock_path("cuda:0").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.robot == "black":
      teacher, cnn, gru = checkpoint.initial_models("cuda:0")
    else:
      teacher, cnn, gru = checkpoint.initial_models(191119)
      teacher, cnn, gru = [module.cuda() for module in (teacher, cnn, gru)]
    env = runtime.NativeEnv(32, 191119, "cuda:0", args.out)
    try:
      depth = env.frames.depth[:2].clone()
      history = env.obs["history"][:2].reshape(2, *teacher.history_shape).clone()
      geometry = env.obs["geometry"][:2].clone()
      cid = env.plant.sim.mj_model.camera("shared_actor_depth").id
      matrices = wp.to_torch(env.plant.sim.wp_data.cam_xmat)[:, cid]
      original = matrices.clone()
      matrices.reshape(32, 3, 3).copy_(
        torch.diag(torch.tensor([1.0, -1.0, -1.0], device="cuda:0"))
      )
      env.frames = AlignedFrames(32, "cuda:0")
      env.obs = env.bind(env.last_native)
      assert env.frames.valid.all() and not env.frames.has_hit.any()
      assert not env.frames.depth.any() and not env.obs["geometry"].any()
      sky_count = int(env.frames.valid.sum())
      matrices.copy_(original)
      env.frames = AlignedFrames(32, "cuda:0")
      env.obs = env.bind(env.last_native)
      assert env.frames.valid.all() and env.frames.has_hit.any()
      with torch.no_grad():
        env.step(
          teacher(
            env.obs["history"].reshape(32, *teacher.history_shape), env.obs["geometry"]
          )
        )
    finally:
      env.close()

    student = SharedStudent(teacher, cnn, gru)
    with torch.no_grad():
      gru.projection.weight.normal_(std=0.03)
    identity = checkpoint.teacher_identity(teacher)
    before_cnn, before_gru = (
      tensor_digest(cnn.state_dict()),
      tensor_digest(gru.state_dict()),
    )
    state = None
    rows, depths, histories, handoffs = [], [], [], []
    for tick in range(541):
      episode = torch.tensor([int(tick >= 135) + int(tick >= 410), 0], device="cuda:0")
      row = dict(
        tick=tick,
        episode=episode,
        capture_episode=episode,
        frame=torch.full((2,), tick // 5, dtype=torch.long, device="cuda:0"),
        capture_tick=torch.full((2,), tick // 5 * 5, dtype=torch.long, device="cuda:0"),
        mask=torch.full((2,), not 265 <= tick < 280, dtype=torch.bool, device="cuda:0"),
      )
      old_count = None if state is None else state.fresh_count.clone()
      # tick540在真实Adam前保留一条新帧计算图，其余为采样。
      with torch.set_grad_enabled(tick == 540):
        _, visual, state = student(history, depth, row, state)
      if old_count is not None and state.fresh_count[1] < old_count[1]:
        handoffs.append(tick)
      rows.append(
        {
          name: value.cpu() if torch.is_tensor(value) else value
          for name, value in row.items()
        }
      )
      depths.append(depth.cpu())
      histories.append(history)
    assert len(handoffs) >= 3 and state.fresh_count.max() <= 50
    loss, _ = distillation_loss(history, geometry, visual, student, teacher, 1.0)
    optimizer = torch.optim.Adam(student.student_parameters(), lr=3e-4)
    loss.backward()
    assert all(
      p.grad is not None and torch.isfinite(p.grad).all()
      for p in student.student_parameters()
    )
    optimizer.step()
    cache = EpisodePrefix(2, 6000, gru.max_age)
    sample = dict(depth=torch.stack(depths), metadata=rows)
    cache.extend(sample)
    refreshed = cache.refresh(student, state.detach(), "cuda:0")
    gpu_rows = [
      {
        name: value.cuda() if torch.is_tensor(value) else value
        for name, value in row.items()
      }
      for row in rows
    ]
    with torch.no_grad():
      _, _, reference = student.forward_sequence(
        torch.stack(histories), sample["depth"].cuda(), gpu_rows
      )
    torch.testing.assert_close(refreshed.hidden, reference.hidden, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
      refreshed.warm_hidden, reference.warm_hidden, rtol=2e-5, atol=2e-6
    )
    assert checkpoint.teacher_identity(teacher) == identity
    assert tensor_digest(cnn.state_dict()) != before_cnn
    assert tensor_digest(gru.state_dict()) != before_gru
    assert checkpoint.source_identity() == source
    result = dict(
      robot=args.robot,
      source_identity=source,
      passed=True,
      test_only=True,
      real_physics_controls=1,
      real_gpu_sky_empty_healthy_lanes=sky_count,
      synthetic_memory_stream=True,
      synthetic_controls=541,
      optimizer_calls=1,
      handoff_control_ticks_lane1=handoffs,
      cache_capacity=cache.capacity,
      retained_frames_per_lane=cache.length.tolist(),
      final_fresh_count=state.fresh_count.tolist(),
      max_hidden_error=float((refreshed.hidden - reference.hidden).abs().max()),
      max_warm_hidden_error=float(
        (refreshed.warm_hidden - reference.warm_hidden).abs().max()
      ),
      frozen_teacher_unchanged=True,
      cnn_and_gru_changed=True,
      capability_qualified=False,
    )
    (args.out / "CHECK.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
  main()
