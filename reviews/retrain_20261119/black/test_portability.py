"""新包不依赖原工程目录或同名顶层模块。"""

from pathlib import Path

from support.black.config import TASKS, make_general_configs
from support.black.parameters import PARAMETERS
from support.black.specialty_env import make_specialty_configs

import black.model
import black.runtime


def test_explicit_package_and_local_sources():
  root = Path(black.runtime.__file__).resolve().parents[1]
  assert Path(black.model.__file__).resolve().is_relative_to(root)
  assert black.runtime.ROOT == root


def test_general_and_all_six_specialty_configs():
  cfg, _ = make_general_configs(32)
  assert len(cfg.scene.terrain.terrain_generator.sub_terrains) == 7
  for task in TASKS:
    cfg, _ = make_specialty_configs(task, num_envs=32, course_parameters=PARAMETERS)
    assert cfg.commands["twist"].task == task
    assert cfg.scene.num_envs == 32
