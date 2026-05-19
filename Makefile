.DEFAULT_GOAL := help
.PHONY: setup fmt lint test ci backtest figures paper compile-paper docs help

setup: ## Install deps + pre-commit hooks
	uv sync --all-extras
	uv run pre-commit install

fmt: ## Format code
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint: ## Lint + typecheck
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run black --check src tests
	uv run mypy src

test: ## Run tests with coverage
	uv run pytest --cov=tessera --cov-report=term-missing

ci: fmt lint test ## Run full CI pipeline (fmt → lint → test); must be green before commit

backtest: ## Run walk-forward backtest; outputs to data/backtest_runs/
	uv run tessera backtest

figures: ## Regenerate all paper + docs figures from data
	uv run python paper/figures/generate_all.py

paper: ## Run live paper-trade (Binance Testnet + Bybit Demo)
	uv run tessera paper start --config configs/live.yaml

compile-paper: ## Compile paper/main.tex → paper/main.pdf (requires tectonic)
	@command -v tectonic >/dev/null 2>&1 || \
	  { echo "ERROR: tectonic not found. Install: brew install tectonic  |  cargo install tectonic"; exit 1; }
	cd paper && tectonic main.tex
	@echo "Output: paper/main.pdf"

docs: ## Serve MkDocs docs locally at http://localhost:8000
	uv run mkdocs serve

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
