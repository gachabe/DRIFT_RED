PY ?= python

.PHONY: setup data check verify demo mini full test

setup:
	uv sync

data:
	$(PY) -m src.data --download

check:
	$(PY) -m src.validation

verify:
	$(PY) -m src.validation --run-tests

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m experiments.04_final --config configs/demo.yaml

mini:
	$(PY) -m experiments.04_final --config configs/mini.yaml

full:
	$(PY) -m experiments.04_final --config configs/full.yaml
