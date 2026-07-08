.PHONY: ci check lint format test install

ci: format lint check test

check:
	uv run pyright

lint:
	uvx ruff check --fix

format:
	uvx ruff format

test:
	uv run pytest .
	uv run xdoctest robokit

install:
	uv sync --all-extras
