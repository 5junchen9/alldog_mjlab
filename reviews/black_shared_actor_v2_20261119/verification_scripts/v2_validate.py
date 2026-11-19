"""Run the candidate-scoped checks and preserve their exact commands and output."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/wujuncheng/Documents/Codex/2026-11-11/task/black_rough_v2')
REL = Path('preparation/rough_v2/shared_actor_v2_20261119')
OUT = ROOT / REL / 'evidence/final_validation'
OUT.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
env['PATH'] = '/home/wujuncheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:' + env['PATH']
base = ['uv', 'run', '--no-sync', '--offline']
config = json.loads((ROOT / REL / 'pyrightconfig.json').read_text())
ty = ['ty', 'check', str(REL), '--config', f'src.include=["{REL}"]']
for path in config['extraPaths']:
    ty += ['--extra-search-path', str((REL / path))]
commands = {
    'pytest': ['pytest', str(REL), '-q'],
    'ruff': ['ruff', 'check', str(REL)],
    'format': ['ruff', 'format', '--check', str(REL)],
    'ty': ty,
    'pyright': ['pyright', '-p', str(REL / 'pyrightconfig.json')],
    'plan': ['python', str(REL / 'run.py')],
}
results = {}
for name, args in commands.items():
    start = time.time()
    process = subprocess.run(base + args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / (name + '.log')).write_bytes(process.stdout)
    results[name] = dict(command=base+args, exit_code=process.returncode, seconds=round(time.time()-start, 3),
                         log_sha256=hashlib.sha256(process.stdout).hexdigest())
    print(name, process.returncode, flush=True)
(OUT / 'RESULT.json').write_text(json.dumps(results, indent=2) + '\n')
raise SystemExit(any(row['exit_code'] for row in results.values()))
