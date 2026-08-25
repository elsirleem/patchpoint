.PHONY: demo test lint typecheck

demo:
	python -m patchpoint.cli demo

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy patchpoint
