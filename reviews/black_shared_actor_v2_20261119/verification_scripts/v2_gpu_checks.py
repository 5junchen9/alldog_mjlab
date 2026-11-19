"""固定32环境的接线验证，不授予教师资格，不提供正式训练入口。"""
import argparse
import fcntl
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/home/wujuncheng/Documents/Codex/2026-11-11/task/black_rough_v2')
CANDIDATE = ROOT / 'preparation/rough_v2/shared_actor_v2_20261119'
sys.path.insert(0, str(CANDIDATE))
from checkpoint import initial_models, load_bundle, tensor_digest, teacher_identity
from eval_collect import source_files
from eval_protocol import digest
from model import make_student_modules
from runtime import NativeEnv
from student import SharedStudent
from student_training import train_student
from export import export_student

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('case', choices=['reset', 'student', 'student_resume'])
parser.add_argument('out', type=Path)
parser.add_argument('--bundle', type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
env = None
source_before = source_files()
with Path('/tmp/mjlab_black_matrix_1000_cuda_0.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        torch.manual_seed(119111)
        torch.set_num_threads(2)
        packet = None
        if args.bundle:
            teacher, cnn, gru, packet = load_bundle(args.bundle, 'cuda:0')
        else:
            teacher, cnn, gru = initial_models('cuda:0')
        if args.case == 'student':
            cnn, gru = make_student_modules()
            cnn, gru = cnn.to('cuda:0'), gru.to('cuda:0')
        before = tensor_digest(teacher.state_dict())
        visual_before = {'cnn': tensor_digest(cnn.state_dict()), 'gru': tensor_digest(gru.state_dict())}
        env = NativeEnv(32, 119111, 'cuda:0', args.out)
        env.policy_identity = teacher_identity(teacher)
        command = env.plant.command_manager.get_term('twist')
        assert bool(command.flat_mask.all())
        assert bool((env.plant.scene.terrain.terrain_types == 0).all())
        if args.case == 'reset':
            student = SharedStudent(teacher, cnn, gru).eval()
            hidden = None
            with torch.no_grad():
                for _ in range(2):
                    action, _, hidden = student(env.obs['history'].reshape(32, 5, 45), env.frames.depth, env.frames.metadata(), hidden)
                    _, _, done, _ = env.step(action)
                    assert not bool(done.any())
                old_frame, old_depth = env.frames.frame.clone(), env.frames.depth.clone()
                env.plant.episode_length_buf[0] = env.plant.max_episode_length - 1
                action, _, hidden = student(env.obs['history'].reshape(32, 5, 45), env.frames.depth, env.frames.metadata(), hidden)
                obs, _, done, info = env.step(action)
                assert done.nonzero().flatten().tolist() == [0]
                assert info['time_outs'][0] and not info['terminated'][0]
                assert not torch.equal(info['terminal_history'][0], obs['history'][0].reshape(5, 45))
                assert not torch.equal(info['terminal_critic'][0], obs['critic'][0])
                assert torch.equal(env.frames.frame[1:], old_frame[1:])
                assert torch.equal(env.frames.depth[1:], old_depth[1:])
                assert env.frames.capture_tick[0] == 3
                history = obs['history'].reshape(32, 5, 45)
                torch.testing.assert_close(history[0], history[0, -1:].expand(5, 45), rtol=0, atol=0)
                _, _, after = student(history, env.frames.depth, env.frames.metadata(), hidden)
                _, _, fresh = student(history, env.frames.depth, env.frames.metadata(), None)
                torch.testing.assert_close(after.hidden[0], fresh.hidden[0], rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(after.hidden[1:], hidden.hidden[1:], rtol=0, atol=0)
            result = dict(controls=3, optimizer_calls=0, reset_ids=[0], terminal_targets_before_reset=True,
                          native_timeout_injected=True, camera_history_gru_isolation=True)
        else:
            result = train_student(env, teacher, cnn, gru, 1, args.out,
                                   resume=packet if args.case == 'student_resume' else None, save_every=1)
            result['visual_changed'] = {'cnn': visual_before['cnn'] != tensor_digest(cnn.state_dict()),
                                       'gru': visual_before['gru'] != tensor_digest(gru.state_dict())}
            assert all(result['visual_changed'].values())
            result['export'] = export_student(SharedStudent(teacher, cnn, gru), args.out / 'student_jit.pt')
        assert tensor_digest(teacher.state_dict()) == before
        assert source_files() == source_before
        result.update(case=args.case, environments=32, test_only=True, teacher_qualification_granted=False,
                      formal_student_started=False, teacher_unchanged=True, all_flat=True,
                      source_sha256=digest(source_before), capability_evaluated=False)
        (args.out/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
    except BaseException as error:
        (args.out/'FAILED.json').write_text(json.dumps({'error':repr(error)})+'\n')
        raise
    finally:
        if env is not None:
            env.close()
