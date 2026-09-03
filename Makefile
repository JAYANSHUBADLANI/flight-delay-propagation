.PHONY: help data data-full absorption predict recover figures test all clean

help:
	@echo "make data        download the two months this analysis used (Jan+Feb 2024, ~10 min)"
	@echo "make data-full   download all twelve months of 2024 (~50 min, ~2.8 GB)"
	@echo "make absorption  rebuild rotations and write reports/*.json"
	@echo "make predict     train the as-of-cutoff model, write prediction metrics"
	@echo "make recover     solve the recovery MILP across real disruptions"
	@echo "make figures     render reports/figures/*.png from those JSON files"
	@echo "make test        run the test suite (no data needed)"
	@echo "make all         reproduce this analysis end to end"

data:
	python3 scripts/download_data.py 2024 1 2

data-full:
	python3 scripts/download_data.py 2024

absorption:
	python3 scripts/run_absorption.py

predict:
	python3 scripts/run_prediction.py

recover:
	python3 scripts/run_recovery.py

figures:
	python3 scripts/make_figures.py

test:
	python3 -m pytest

all: data absorption predict recover figures test

clean:
	rm -rf reports/*.json reports/figures/*.png __pycache__ .pytest_cache
