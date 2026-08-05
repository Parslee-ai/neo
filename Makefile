.PHONY: help install install-dev test format lint hooks ci-local clean sync-version

# Prefer the project venv over whatever happens to be on PATH. A system-wide
# `ruff` is routinely a different version from the pinned one, and linting with
# the wrong version is precisely how a green local tree turned main red.
PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

help:
	@echo "Neo Makefile Commands:"
	@echo ""
	@echo "  make install       - Install Neo in editable mode"
	@echo "  make install-dev   - Install with development tools"
	@echo "  make test          - Run all tests"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make sync-version  - Propagate pyproject's version to the derived files"
	@echo "  make hooks         - Enable the pre-commit hook (run once per clone)"
	@echo "  make ci-local      - Run exactly what CI runs, before pushing"
	@echo "  make clean         - Clean up generated files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

format:
	$(PYTHON) -m black src/ tests/

lint:
	$(PYTHON) -m ruff check src/ tests/ tools/

# Bump `version` in pyproject.toml, then run this. The plugin manifests and
# __version__ are derived, never hand-edited.
sync-version:
	$(PYTHON) tools/sync_version.py

hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit hook enabled (.githooks). It runs the same lint + tests as CI."

ci-local:
	$(PYTHON) -m ruff check src/ tests/ tools/
	$(PYTHON) -m pytest -q

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf build dist *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
