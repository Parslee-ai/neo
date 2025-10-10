.PHONY: help install install-dev test format lint clean

help:
	@echo "Neo Makefile Commands:"
	@echo ""
	@echo "  make install       - Install Neo in editable mode"
	@echo "  make install-dev   - Install with development tools"
	@echo "  make test          - Run all tests"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make clean         - Clean up generated files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

format:
	black src/ tests/

lint:
	ruff check src/ tests/

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf build dist *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
