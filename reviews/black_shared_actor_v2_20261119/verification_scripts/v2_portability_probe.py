"""Read-only CPU import/config probe for the copied Black tree."""

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
original = Path('/home/wujuncheng/Documents/Codex/2026-11-11/task/black_rough_v2')
candidate = root / 'preparation/rough_v2/shared_actor_v2_20261119'
opened = set()

def audit(event, args):
  if event == 'open' and isinstance(args[0], (str, bytes)):
    name = Path(args[0]).resolve()
    if root != original and name.is_relative_to(original):
      raise RuntimeError(f'ORIGINAL_TREE_ACCESS_FORBIDDEN: {name}')
    if name.is_relative_to(root):
      opened.add(str(name.relative_to(root)))

sys.addaudithook(audit)
sys.path[:0] = [str(candidate), str(root / 'src')]
result = {'root': str(root), 'gpu_requested': False}
try:
  import torch
  torch.set_num_threads(1)
  def no_weight_load(*args, **kwargs):
    raise RuntimeError('CHECKPOINT_LOAD_FORBIDDEN_IN_SCRATCH_IMPORT_PROBE')
  torch.load = no_weight_load
  import run
  from checkpoint import initial_models
  initial_models('cpu')
  result['scratch_modules_without_checkpoint'] = True
  import runtime
  runtime.native_imports()
  for name in ('camera_contract', 'chunk_renderer_fixed', 'direct160', 'parameters',
               'passage_course', 'specialty_env'):
    importlib.import_module(name)
  from specialty_env import make_specialty_configs
  from parameters import PARAMETERS
  from passage_course import POLICY
  cfg, _ = make_specialty_configs('stairs', num_envs=32, seed=119032,
                                 course_parameters=PARAMETERS, passage_policy=POLICY)
  from curriculum import configure
  configure(cfg, 'flat_foundation')
  result['config_without_environment'] = True
  from mjlab.asset_zoo.robots.black_stock.black_constants import get_spec
  spec = get_spec()
  result['black_spec_parse'] = {'bodies': len(spec.bodies), 'geoms': len(spec.geoms)}
  from eval_collect import source_files
  from eval_protocol import digest
  identity = source_files()
  result['source_identity'] = identity
  result['source_sha256'] = digest(identity)
  result['passed'] = True
except BaseException as error:
  import traceback
  result.update(passed=False, error=repr(error), traceback=traceback.format_exc())
finally:
  result['cuda_initialized'] = bool('torch' in sys.modules and torch.cuda.is_initialized())
  result['local_modules'] = {
    name: str(Path(module.__file__).resolve().relative_to(root))
    for name, module in sorted(sys.modules.items())
    if getattr(module, '__file__', None)
    and Path(module.__file__).is_file()
    and Path(module.__file__).resolve().is_relative_to(root)
  }
  result['source_outside_copy'] = {
    name: str(Path(module.__file__).resolve())
    for name, module in sorted(sys.modules.items())
    if getattr(module, '__file__', None)
    and Path(module.__file__).is_file()
    and Path(module.__file__).resolve().is_relative_to(original)
    and not Path(module.__file__).resolve().is_relative_to(root)
  }
  result['opened_project_files'] = sorted(opened)
  result['search_paths'] = sys.path
  result['installed'] = {
    name: importlib.metadata.version(name)
    for name in ('torch', 'mujoco', 'mujoco-warp', 'warp-lang', 'rsl-rl-lib',
                 'numpy', 'tensordict', 'mjlab', 'onnx', 'onnxruntime')
  }
  output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
  print(json.dumps({k: result[k] for k in ('passed', 'cuda_initialized', 'installed')}, indent=2))
  if not result['passed']:
    print(result['traceback'])
