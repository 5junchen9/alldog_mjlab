.PHONY: test check

REVIEW_FILES = black wolf support/black/sparse_support.py support/black/test_sparse_support.py \
 support/black/causal_visual_gru.py support/black/config.py support/black/critic.py \
 support/black/depth_him.py support/black/exit_stability.py support/black/him_base.py \
 support/black/progress_credit.py support/black/specialty_env.py support/black/stair_metrics.py \
 src/mjlab/rl/wolf_him/estimator.py \
 src/mjlab/tasks/velocity/config/black_actuation/actuation.py \
 src/mjlab/tasks/velocity/config/black_rough/mdp.py src/mjlab/tasks/velocity/mdp/rewards.py

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --offline python -m pytest -q black wolf support/black/test_sparse_support.py

check: test
	uv run --no-sync --offline python -m ruff check $(REVIEW_FILES)
	uv run --no-sync --offline python -m ruff format --check $(REVIEW_FILES)
