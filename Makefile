PT ?= pt-cu126

.PHONY: ci check lint format test test-all install

ci: format lint check test-all

check:
	uv run --extra $(PT) pyright

lint:
	uvx ruff check --fix

format:
	uvx ruff format

test:
	uv run pytest -m "not torch"
	uv run xdoctest robokit

test-all:
	uv run --extra $(PT) --extra mjcf pytest .
	uv run --extra $(PT) xdoctest robokit

install:
	uv sync --extra $(PT)
