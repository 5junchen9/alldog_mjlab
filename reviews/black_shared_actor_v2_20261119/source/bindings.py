"""v2私有配置绑定；依赖随项目根搬移，不读取旧模型权重。"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREP = ROOT / "preparation/rough_v2"
FROZEN446 = PREP / "terrain446_success12335_20261112"
sys.path.insert(1, str(FROZEN446))

from terrain446_models import (  # noqa: E402
  SCHEMA,
  Teacher72,
  TerrainModel,
  TerrainReservoir,
)
from terrain446_observations import validate_raw446  # noqa: E402
from terrain446_runtime import TerrainDaggerPPO  # noqa: E402
from terrain446_runtime import make_configs as comprehensive_configs  # noqa: E402

__all__ = [
  "SCHEMA",
  "TerrainModel",
  "TerrainReservoir",
  "Teacher72",
  "validate_raw446",
  "TerrainDaggerPPO",
  "comprehensive_configs",
]

TASKS = ("stairs", "platform", "high_wall", "slope", "broken_bridge", "plum_stakes")
LAMBDA_FLAT = 4.0
T1_SHA = "55be4aff5a54232a45f27e6df567d0de49d53e4aeb9096c1e96c307b343749f7"
