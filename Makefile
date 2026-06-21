# Spectral-saturation paper — task runner.
#
# Targets:
#   make install  — pip-install the package with every optional extra
#   make data     — fetch every dataset and CLIP-feature cache
#   make results  — run every analysis driver end-to-end
#   make figures  — emit the manuscript figure set
#   make paper    — copy cached JSONs into paper/evidence/ for the manuscript
#   make all      — install + data + results + figures + paper

.PHONY: all install data results figures paper clean help

# Path to the orchestrator, relative to this Makefile.
ORCHESTRATOR    := python run_all_experiments.py
PYTHON          := python

help:
	@echo "Targets:"
	@echo "  make install   — pip install this repo with the optional CLIP / CIFAR / figure extras"
	@echo "  make data      — populate data/ with every dataset (and CLIP features if the cache is cold)"
	@echo "  make results   — run the full results/ pipeline (caching is honoured)"
	@echo "  make figures   — regenerate figures/ from the cached results"
	@echo "  make paper     — copy canonical JSONs into paper/evidence/ for the manuscript"
	@echo "  make all       — install + data + results + figures"
	@echo "  make clean     — remove cached results (forces a full re-run)"

install:
	pip install -e ".[all]"

# On-demand data fetch.  Each driver triggers its own data load on
# demand; this target just pre-creates the data/ tree so reviewers
# can verify the caching behaviour without surprise directory
# creation.
data:
	mkdir -p data
	$(PYTHON) -c "from src.datasets import load_all_datasets; load_all_datasets('data')"
	$(PYTHON) -c "from src.datasets import load_cifar10; load_cifar10()"

results:
	$(ORCHESTRATOR) --skip-figures

figures:
	$(ORCHESTRATOR) --only-figures

paper:
	mkdir -p paper/evidence
	cp -n results/all_31_results.json            paper/evidence/ || true
	cp -n results/nway_pca_results.json          paper/evidence/ || true
	cp -n results/clip_vitb32_binary_results.json paper/evidence/ || true
	cp -n results/clip_vitl14_binary_results.json paper/evidence/ || true
	cp -n results/multistat_results.json         paper/evidence/ || true
	cp -n results/tau_transfer_report.json       paper/evidence/ || true
	cp -n results/active_learning_results.json   paper/evidence/ || true
	@echo "Cited JSONs mirrored to paper/evidence/."

all: install data results figures paper

clean:
	rm -f results/*.json
	@echo "Cached results deleted — next make results will recompute."
