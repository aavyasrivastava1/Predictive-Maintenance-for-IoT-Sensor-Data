.PHONY: setup venv train infer pipeline test clean notebook

PYTHON := .venv/bin/python
PIP := .venv/bin/pip

setup: venv

venv:
	@bash scripts/setup_venv.sh

train: venv
	$(PYTHON) pipelines/local_train.py --model both

infer: venv
	$(PYTHON) pipelines/local_inference.py

pipeline: venv
	$(PYTHON) pipelines/azure/train_pipeline.py

test: venv
	$(PYTHON) -m pytest tests/ -v

notebook: venv
	$(PYTHON) -m jupyter notebook notebooks/

clean:
	rm -rf data/raw data/processed models/* artifacts/* logs/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
