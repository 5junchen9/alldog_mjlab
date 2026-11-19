"""CPU验证私有绑定优先、随机初始化与配置不读取旧权重。"""

import importlib.machinery
import os
import subprocess
import sys
from pathlib import Path

import runtime


def test_native_imports_selects_the_candidate_binding(monkeypatch):
  monkeypatch.setattr(sys, "path", sys.path.copy())
  runtime.native_imports()
  spec = importlib.machinery.PathFinder.find_spec("bindings", sys.path)
  assert spec is not None and spec.origin is not None
  assert Path(spec.origin) == runtime.HERE / "bindings.py"


def test_cpu_scratch_and_configuration_do_not_load_checkpoint():
  source = """
import sys
from pathlib import Path
here = Path(sys.argv[1])
sys.path[:0] = [str(here), str(here.parents[2] / 'src')]
import torch
torch.set_num_threads(1)
def forbidden(*args, **kwargs):
  raise AssertionError('scratch/config must not load any checkpoint')
torch.load = forbidden
from checkpoint import initial_models
initial_models('cpu')
import runtime
runtime.native_imports()
import bindings
assert Path(bindings.__file__) == here / 'bindings.py'
assert bindings.ROOT == here.parents[2]
from specialty_env import make_specialty_configs
from parameters import PARAMETERS
from passage_course import POLICY
from curriculum import configure
cfg, _ = make_specialty_configs('stairs', num_envs=32, seed=119032,
    course_parameters=PARAMETERS, passage_policy=POLICY)
configure(cfg, 'flat_foundation')
assert not torch.cuda.is_initialized()
print('scratch/config/import passed without checkpoint or CUDA')
"""
  result = subprocess.run(
    [sys.executable, "-c", source, str(runtime.HERE)],
    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
    capture_output=True,
    text=True,
    timeout=60,
  )
  assert result.returncode == 0, result.stdout + result.stderr
  assert "passed without checkpoint or CUDA" in result.stdout
