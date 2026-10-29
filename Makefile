# Cierra: one entry point for setup, tests, builds, evaluations and the local LLM.
#
#   make setup          virtual environment with the pinned versions (requirements-dev.txt) and Cierra itself
#   make test           the test suite (no network, no GPU, no model needed)
#   make build          compile every look in looks/ to DCTL, WGSL, CUDA, CLF / .cube LUTs and ASC CDL (out/build/)
#   make pipeline-demo  a CC0 camera RAW still -> a keyframed Cierra program -> a ProRes-compatible movie
#   make eval           score the deterministic rules driver + tuner on both held-out sets (free, no model)
#   make llm-setup      build llama.cpp (CPU) and download an open-weight model, both pinned and hash-checked
#   make eval-llm       the same evaluations with the local model (start it with make llm-start)
#   make help           every target
#
# Variables: PYTHON (interpreter for the venv, default python3), VENV (default .venv), MODEL (7b | 1.5b),
# LLM_THREADS (CPU threads for the model server, default 8), LABEL (name of an evaluation run).

PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
MODEL ?= 7b
LLM_THREADS ?= 8
LABEL ?= $(shell date -u +%Y%m%d)
LOOKS := $(wildcard looks/*.cierra)
EXAMPLES := $(wildcard examples/*.cierra)
LLM_ENV = CIERRA_LLM_URL=http://127.0.0.1:8090/v1 CIERRA_MODEL=$(shell bash tools/setup_llm.sh alias $(MODEL)) \
          CIERRA_LLM_TIMEOUT=3600

.PHONY: help setup test test-fast check build docs targets sbom eval eval-heldout eval-freeform pipeline-demo \
        llm-setup llm-start llm-stop llm-status eval-llm loop-demo clean

help:  ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

setup: $(PY)  ## create the venv with pinned dependencies and install Cierra (editable)

test: $(PY)  ## run the whole test suite
	$(PY) -m pytest

test-fast: $(PY)  ## the language, checker and engine tests only (seconds)
	$(PY) -m pytest tests/test_checker.py tests/test_conformance.py tests/test_engine.py tests/test_grammar.py

check: $(PY)  ## check every look and example
	@for f in $(LOOKS) $(EXAMPLES); do echo "== $$f"; $(PY) -m cierra check $$f > /dev/null && echo ok || exit 1; done

build: $(PY)  ## compile every look to DCTL, WGSL, CUDA (.cu), LUTs (.cube / .clf) and CDL where it allows, into out/build/
	@for f in $(LOOKS); do \
	  n=$$(basename $$f .cierra); echo "== $$n"; \
	  $(PY) -m cierra dctl --split $$f out/build/$$n || exit 1; \
	  $(PY) -m cierra wgsl --split $$f out/build/$$n || exit 1; \
	  $(PY) -m cierra cuda --split $$f out/build/$$n || exit 1; \
	  $(PY) -m cierra lut $$f out/build/$$n 2>/dev/null || echo "   no LUT: the look has spatial ops (DCTL parts only)"; \
	  $(PY) -m cierra cdl $$f out/build/$$n 2>/dev/null || echo "   no CDL: the look uses ops a CDL cannot hold"; \
	done

docs: $(PY)  ## regenerate docs/authoring.md and docs/ops_manifest.json from the op table
	$(PY) tools/make_authoring.py

targets: $(PY)  ## ΔE00 of every compiled target against the reference engine (docs/targets.json)
	$(PY) tools/target_report.py

sbom: $(PY)  ## dependency licences (docs/sbom.json); fails on GPL / AGPL / non-commercial in the library
	$(PY) tools/sbom.py

# A CC0 sample from raw.pixls.us (Pentax K-5 II, 14-bit DNG), pinned by its published SHA-256
SAMPLE_RAW := cache/demo/pentax_k5ii.dng
SAMPLE_URL := https://raw.pixls.us/getfile.php/873/nice/Pentax%20-%20K-5%20II%20-%2014bit%20%283%3A2%29.DNG
SAMPLE_SHA := aa41f0d9ab8d3e21e468b4c61d4d1ddae6a4fbe37f629b78159246cbb94dd56b

pipeline-demo: $(PY)  ## RAW still -> examples/tungsten_fix.cierra -> out/pipeline/tungsten_fix.mov (needs ffmpeg)
	@mkdir -p cache/demo
	@[ -f $(SAMPLE_RAW) ] || curl -fL --retry 3 -o $(SAMPLE_RAW) "$(SAMPLE_URL)"
	@echo "$(SAMPLE_SHA)  $(SAMPLE_RAW)" | sha256sum -c -
	$(PY) tools/raw_to_video.py examples/tungsten_fix.cierra $(SAMPLE_RAW) --wb daylight --frames 72 --width 1280 	  --out out/pipeline/tungsten_fix.mov

eval: eval-heldout eval-freeform  ## rules driver + tuner on both held-out sets (deterministic, free)

eval-heldout: $(PY)
	$(PY) tools/eval_heldout.py --mode tuner --driver rules --label rules_tuner_$(LABEL)

eval-freeform: $(PY)
	$(PY) tools/eval_heldout.py --set freeform --mode tuner --driver rules --label rules_tuner_freeform_$(LABEL)

llm-setup:  ## build llama.cpp (CPU) and download MODEL (7b | 1.5b) into llm/, pinned and hash-checked
	bash tools/setup_llm.sh all $(MODEL)

llm-start:  ## start the local model server (127.0.0.1:8090, CPU, LLM_THREADS threads)
	CIERRA_LLM_THREADS=$(LLM_THREADS) CIERRA_GGUF=$$(bash tools/setup_llm.sh path $(MODEL)) \
	  CIERRA_GGUF_ALIAS=$$(bash tools/setup_llm.sh alias $(MODEL)) bash tools/llm_server.sh start

llm-stop:  ## stop the local model server
	bash tools/llm_server.sh stop

llm-status:
	bash tools/llm_server.sh status

eval-llm: $(PY)  ## held-out loop and free-form tuner with the local model (CPU: tens of minutes)
	$(LLM_ENV) $(PY) tools/eval_heldout.py --mode loop --driver openweight --label $(MODEL)_loop_$(LABEL)
	$(LLM_ENV) $(PY) tools/eval_heldout.py --set freeform --mode tuner --structure none --driver openweight \
	  --label $(MODEL)_tuner_freeform_$(LABEL)

loop-demo: $(PY)  ## one grading-loop run on a test frame with whatever driver the environment selects
	$(PY) -m cierra loop "warmer faces, a little more contrast, keep the blacks clean" \
	  tests/data/frames/crowd.npy out/loop_demo

clean:  ## remove build and evaluation outputs (keeps the venv and llm/)
	rm -rf out/build out/evals out/loop_demo out/pipeline cache/pytest
