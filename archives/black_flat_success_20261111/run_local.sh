#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (($# == 0)); then
  printf '%s\n' 'No default job. Read README.zh-CN.md before running an explicit command.'
  exit 0
fi
# Local overrides permit another machine to select its existing offline environment.
BLACK_UV="${BLACK_UV:-/snap/astral-uv/1781/bin/uv}"
BLACK_PYTHON="${BLACK_PYTHON:-/home/wujuncheng/rl_train/wolf_env_20261109/venv/bin/python}"
cd "$ROOT"
unset PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src"
export UV_CACHE_DIR="${BLACK_UV_CACHE_DIR:-/tmp/black_archive_uv}"
export MPLCONFIGDIR="${BLACK_MPLCONFIGDIR:-/tmp/black_archive_matplotlib}"
exec "$BLACK_UV" run --offline --no-project --python "$BLACK_PYTHON" "$@"
