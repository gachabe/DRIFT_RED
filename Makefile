PY ?= python

.PHONY: setup data check verify demo mini full test revision revision-demo consistency

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

revision:
	$(PY) -m experiments.run_all --config configs/full.yaml

revision-demo:
	$(PY) -m experiments.run_all --config configs/demo.yaml --skip-tests

consistency:
	$(PY) -m src.consistency --results results/full --reference reference/manuscript_numbers.csv
