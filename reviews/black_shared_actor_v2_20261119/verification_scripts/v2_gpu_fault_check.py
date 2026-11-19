"""Injected-state diagnostic check. This does not replay or explain the old NaN."""
import fcntl
import json
import sys
from pathlib import Path

import torch
import warp as wp

ROOT = Path('/home/wujuncheng/Documents/Codex/2026-11-11/task/black_rough_v2')
CANDIDATE = ROOT / 'preparation/rough_v2/shared_actor_v2_20261119'
sys.path.insert(0, str(CANDIDATE))
from checkpoint import initial_models, teacher_identity
from eval_collect import source_files
from eval_protocol import digest
from fault_capture import write_fault
from runtime import NativeEnv

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
source = source_files()
env = None
with Path('/tmp/mjlab_black_matrix_1000_cuda_0.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        torch.manual_seed(119113)
        torch.set_num_threads(2)
        teacher, _, _ = initial_models('cuda:0')
        env = NativeEnv(32, 119113, 'cuda:0', out)
        env.policy_identity = teacher_identity(teacher)
        with torch.no_grad():
            action = teacher(env.obs['history'].reshape(32, 5, 45), env.obs['geometry'])
        wp.to_torch(env.plant.sim.wp_data.qacc_warmstart)[0, 0] = float('nan')
        error = None
        try:
            env.step(action)
        except FloatingPointError as caught:
            error = repr(caught)
        assert error is not None
        fault = env.probe.fault
        assert fault is not None and fault['bad_lanes'].tolist() == [0]
        assert fault['stage'].endswith(':before') and not fault['replayable']
        assert fault['context']['policy_identity'] == env.policy_identity
        assert 'qacc_warmstart' in fault['pre']
        assert 'action_history' in fault['context']
        packet_path = write_fault(env.probe, out / 'first_fault')
        packet = torch.load(packet_path, map_location='cpu', weights_only=False)
        assert packet['bad_lane_count'] == 1
        assert source_files() == source
        result = dict(test_only=True, injected_field='qacc_warmstart', injected_lane=0,
                      stage=fault['stage'], source_sha256=digest(source), optimizer_calls=0,
                      error=error, fields=list(fault['pre']), atomic_packet_loaded=True,
                      natural_nan_reproduced=False, original_root_cause_established=False,
                      replayable=False, policy_identity_saved=True)
        (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    finally:
        if env is not None:
            env.close()
