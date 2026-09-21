.PHONY: install dev run test lint

install:
	python -m pip install -e .

dev:
	python -m pip install -e '.[dev]'

run:
	meeting-stt

test:
	pytest

lint:
	ruff check .

