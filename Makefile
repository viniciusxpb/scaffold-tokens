PYTHON ?= micromamba run -n scaffold python3

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-18s %s\n", $$1, $$2}'

# --- Setup ---

setup: ## Create env and install all dependencies
	@command -v micromamba >/dev/null 2>&1 || { echo "micromamba not found. Install: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"; exit 1; }
	micromamba create -n scaffold python=3.10 -c conda-forge -y
	micromamba run -n scaffold pip install -r requirements.txt

# --- Dataset ---

download: ## Download dataset from Hugging Face
	$(PYTHON) utils/download_dataset.py

download-model: ## Download pre-trained model from Hugging Face
	$(PYTHON) utils/download_model.py

validate: ## Validate shard integrity
	$(PYTHON) utils/validate_shards.py

# --- Training ---

sanity: ## Quick sanity check (mini model, 30 steps)
	$(PYTHON) sanity.py

train: ## Train GPT-2 124M (~55 min on RTX 3060)
	$(PYTHON) train.py

# --- Inference ---

generate: ## Generate 100 words (no prompt)
	$(PYTHON) inference.py --words 100

generate-prompt: ## Generate with prompt (PROMPT="...")
	$(PYTHON) inference.py --words 100 --prompt "$(PROMPT)"

generate-tokens: ## Generate showing <ff_N> scaffold tokens
	$(PYTHON) inference.py --words 100 --show-tokens

test-prompts: ## Run 53 test prompts with analysis
	$(PYTHON) utils/run_test_prompts.py

# --- Utilities ---

clean: ## Delete shards, checkpoints, runs, and logs
	rm -rf data/shards checkpoints runs logs

.PHONY: help setup download download-model validate sanity train generate generate-prompt generate-tokens test-prompts clean
