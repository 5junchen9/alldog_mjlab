.PHONY: test check

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --offline python -m pytest -q black wolf support/black/test_sparse_support.py

check: test
	uv run --no-sync --offline python -m ruff check black wolf support/black/sparse_support.py support/black/test_sparse_support.py
	uv run --no-sync --offline python -m ruff format --check black wolf support/black/sparse_support.py support/black/test_sparse_support.py
