"""Sequential bounded GPU checks on one frozen source identity."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/wujuncheng/Documents/Codex/2026-11-11/task/black_rough_v2')
WORK = Path('/home/wujuncheng/Documents/Codex/2026-11-19/wo-x/work')
C = ROOT / 'preparation/rough_v2/shared_actor_v2_20261119'
sys.path.insert(0, str(C))
from eval_collect import source_files
from eval_protocol import digest

source = source_files()
ev = C / 'evidence'
base = ['uv', 'run', '--no-sync', '--offline', 'python']
teacher = [str(C/'run.py'), '--mode', 'teacher', '--execute', '--course', 'flat_foundation', '--num-envs', '32', '--rounds', '1', '--save-every', '1']
checks = [
    ('gpu_teacher_final002', teacher),
    ('gpu_teacher_resume_final002', teacher+['--resume-learner', str(ev/'gpu_teacher_final002/teacher_update_000001.pt')]),
    ('gpu_reset_final002', [str(WORK/'v2_gpu_checks.py'), 'reset']),
    ('gpu_student_final002', [str(WORK/'v2_gpu_checks.py'), 'student', '--bundle', str(ev/'gpu_teacher_final002/teacher.pt')]),
    ('gpu_student_resume_final002', [str(WORK/'v2_gpu_checks.py'), 'student_resume', '--bundle', str(ev/'gpu_student_final002/student_update_000001.pt')]),
    ('gpu_fault_final001', [str(WORK/'v2_gpu_fault_check.py')]),
]
receipts = []
for name, args in checks:
    command = base + args + (['--out'] if args[0].endswith('/run.py') else []) + [str(ev/name)]
    log = WORK / ('v2_'+name+'.log')
    with log.open('wb') as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    receipts.append(dict(name=name, command=command, exit_code=completed.returncode, source_sha256=digest(source)))
    (ev/'FINAL_GPU_COMMANDS.json').write_text(json.dumps(receipts, indent=2)+'\n')
    print(name, completed.returncode, flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    assert source_files() == source
(ev/'FINAL_SOURCE.json').write_text(json.dumps({'source_sha256':digest(source), 'files':source}, indent=2)+'\n')
