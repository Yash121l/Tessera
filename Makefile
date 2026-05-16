.DEFAULT_GOAL := help
.PHONY: setup fmt lint test backtest paper docs help

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

backtest: ## Run backtest
	uv run tessera backtest

paper: ## Run paper-trade
	uv run tessera paper

docs: ## Serve docs locally
	uv run mkdocs serve

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
