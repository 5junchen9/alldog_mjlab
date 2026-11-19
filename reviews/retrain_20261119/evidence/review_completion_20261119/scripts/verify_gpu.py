"""有限GPU接线/恢复检查，不授予教师资格，不启动长训。"""

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from black.checkpoint import tensor_digest
from black.eval_collect import device_lock_path
import fcntl


def main():
  p = argparse.ArgumentParser()
  p.add_argument('robot', choices=['black', 'wolf'])
  p.add_argument('case', choices=['teacher', 'teacher_resume', 'student', 'student_resume', 'probe', 'fault', 'scale'])
  p.add_argument('--course', default='flat_foundation')
  p.add_argument('--task')
  p.add_argument('--base', type=Path, required=True)
  a = p.parse_args()
  a.base.mkdir(parents=True, exist_ok=True)
  checkpoint = importlib.import_module(a.robot + '.checkpoint')
  runtime = importlib.import_module(a.robot + '.runtime')
  training = importlib.import_module('black.run' if a.robot == 'black' else 'wolf.training')
  source = checkpoint.source_identity()
  identity_path = a.base / 'SOURCE.json'
  if identity_path.exists():
    assert json.loads(identity_path.read_text()) == source, 'source changed during verification'
  else:
    identity_path.write_text(json.dumps(source, indent=2) + '\n')
  label = '_'.join(filter(None, [a.course, a.task, a.case]))
  out = a.base / label
  out.mkdir(exist_ok=False)
  random.seed(119032)
  np.random.seed(119032)
  torch.manual_seed(119032)
  torch.set_num_threads(2)
  env = None
  with device_lock_path('cuda:0').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
      resume = None
      if a.case.endswith('_resume'):
        phase = a.case.removesuffix('_resume')
        teacher, cnn, gru, resume = checkpoint.load_bundle(a.base / '_'.join(filter(None, [a.course, a.task, phase])) / f'{phase}.pt', 'cuda:0')
      elif a.case == 'student':
        teacher, _, _, _ = checkpoint.load_bundle(a.base / f'{a.course}_teacher' / 'teacher.pt', 'cuda:0')
        model = importlib.import_module(a.robot + '.model')
        cnn, gru = model.make_student_modules()
        cnn, gru = cnn.to('cuda:0'), gru.to('cuda:0')
      elif a.robot == 'black':
        teacher, cnn, gru = checkpoint.initial_models('cuda:0')
      else:
        teacher, cnn, gru = checkpoint.initial_models(119032)
        teacher, cnn, gru = [x.to('cuda:0') for x in (teacher, cnn, gru)]
      options = dict(course=a.course, debug_first_fault=a.case == 'fault', training_sensors=True)
      if a.robot == 'black':
        options['task'] = a.task
      elif a.course in importlib.import_module('wolf.curriculum').SPECIALTIES:
        options['max_level'] = 1
      count = 4096 if a.case == 'scale' else 32
      env = runtime.NativeEnv(count, 119032, 'cuda:0', out, **options)
      env.policy_identity = checkpoint.teacher_identity(teacher)
      original_teacher = env.policy_identity
      before_cnn, before_gru = tensor_digest(cnn.state_dict()), tensor_digest(gru.state_dict())
      assert (env.probe is not None) == (a.case == 'fault')
      import warp as wp
      cid = env.plant.sim.mj_model.camera('shared_actor_depth').id
      nominal = torch.as_tensor(env.plant.sim.mj_model.cam_pos[cid].copy(), device='cuda:0')
      mount = wp.to_torch(env.plant.sim.wp_model.cam_pos)[:, cid]
      delta = (mount - nominal).abs()
      assert delta.max() <= 0.002001 and delta.max() > 0
      valid_depth = env.frames.depth[env.frames.depth > 0]
      assert bool(((valid_depth >= 0.2) & (valid_depth <= 4.0)).all())
      result = {}
      if a.case.startswith('teacher'):
        if a.case == 'teacher' and a.course == 'general':
          # 合成非零课程进度，只验证持久化，不作为能力或晋级证据。
          if a.robot == 'black':
            from mjlab.tasks.velocity.config.black_rough.mdp import state
            state(env.plant).type_ceilings[1] = 2
          else:
            env.plant.command_manager.get_term('twist').general_ceilings[1] = 2
        if a.case == 'teacher' and a.robot == 'black' and a.task == 'platform':
          from support.black.specialty_env import specialty_state
          st = specialty_state(env.plant)
          st.height_level, st.angle_stage = 1, 1
        result = training.train_teacher(env, teacher, cnn, gru, 1, out, resume=resume, save_every=1)
        expected = 2 if resume else 1
        assert result['completed_updates'] == expected
        assert result['optimizer_calls'] == expected * 20
        assert result['him_optimizer_calls'] == expected
        assert checkpoint.teacher_identity(teacher) != original_teacher
        assert tensor_digest(cnn.state_dict()) == before_cnn
        assert tensor_digest(gru.state_dict()) == before_gru
        saved = torch.load(out / 'teacher.pt', map_location='cpu', weights_only=True)
        assert saved['learning_state']['curriculum']['course'] == a.course
        if a.robot == 'black' and a.task == 'platform':
          assert saved['learning_state']['curriculum']['height'] == 1
          assert saved['learning_state']['curriculum']['angle'] == 1
          result['synthetic_nonzero_specialty_curriculum_preserved'] = True
        if a.course == 'general':
          key = 'ceilings' if a.robot == 'black' else 'general_ceilings'
          assert saved['learning_state']['curriculum'][key][1] == 2
          result['synthetic_nonzero_curriculum_preserved'] = True
      elif a.case.startswith('student'):
        fn = importlib.import_module('black.student_training').train_student if a.robot == 'black' else training.train_student
        result = fn(env, teacher, cnn, gru, 1, out, resume=resume, save_every=1)
        expected = 2 if resume else 1
        assert result['optimizer_calls'] == expected * 4
        assert checkpoint.teacher_identity(teacher) == original_teacher
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        assert tensor_digest(cnn.state_dict()) != before_cnn
        assert tensor_digest(gru.state_dict()) != before_gru
        result['qualification_gate_bypassed_only_in_test_helper'] = True
      elif a.case == 'fault':
        import warp as wp
        from black.fault_capture import write_fault
        with torch.no_grad():
          action = teacher(env.obs['history'].reshape(count, *teacher.history_shape), env.obs['geometry'])
        wp.to_torch(env.plant.sim.wp_data.qacc_warmstart)[0, 0] = float('nan')
        try:
          env.step(action)
        except FloatingPointError:
          pass
        assert env.probe.fault['bad_lanes'].tolist() == [0]
        path = write_fault(env.probe, out / 'first_fault')
        packet = torch.load(path, weights_only=False, map_location='cpu')
        result = dict(optimizer_calls=0, injected_field='qacc_warmstart', lane=0,
                      captured_stage=packet['stage'], packet_reloaded=True,
                      natural_nan_reproduced=False, original_root_cause_established=False)
      else:
        env.plant.episode_length_buf[:] = env.max_episode_controls - 1
        with torch.no_grad():
          for _ in range(4):
            action = teacher(env.obs['history'].reshape(count, *teacher.history_shape), env.obs['geometry'])
            env.step(action)
        result = dict(optimizer_calls=0, controls=4, num_envs=count,
                      course_state_shape=list(env.obs['course_state'].shape),
                      finite=True, forced_timeout_and_reset=True, default_substep_probe_absent=env.probe is None)
      result.update(robot=a.robot, case=a.case, course=a.course, task=a.task,
                    source_identity=source, test_only=True, capability_qualified=False,
                    camera=env.camera_receipt, gpu_name=torch.cuda.get_device_name(0),
                    peak_torch_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    warp_allocations_in_torch_peak=False, camera_mount_randomization_checked=True)
      assert checkpoint.source_identity() == source
      (out / 'CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
      print(json.dumps(dict(robot=a.robot, case=a.case, course=a.course, status='passed')))
    except BaseException as error:
      (out / 'FAILED.json').write_text(json.dumps({'error': repr(error), **getattr(error, 'failure_details', {})}, indent=2) + '\n')
      raise
    finally:
      if env is not None:
        env.close()


if __name__ == '__main__':
  main()
