UV ?= uv

.DEFAULT_GOAL := help
.PHONY: help sync lock upgrade test test-verbose run clean

help: ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync: ## Install the project and the dev dependencies in .venv
	$(UV) sync --all-groups

lock: ## Refresh uv.lock without touching the environment
	$(UV) lock

upgrade: ## Upgrade the locked dependencies and sync them
	$(UV) sync --all-groups --upgrade

test: ## Run the tests
	$(UV) run pytest -q

test-verbose: ## Run the tests with the full output
	$(UV) run pytest -vv

run: ## Run main.py, calls the case agent with the real model
	$(UV) run python main.py

clean: ## Remove the caches and the build artifacts
	rm -rf .pytest_cache dist build *.egg-info
	find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
