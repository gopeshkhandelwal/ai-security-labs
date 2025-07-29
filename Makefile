# AI Security Labs - Simple Makefile
# ==================================

# Variables
PYTHON := python3
PIP := pip3
VENV := .venv
VENV_BIN := $(VENV)/bin
PYTHON_VENV := $(VENV_BIN)/python
PIP_VENV := $(VENV_BIN)/pip

# Project directories
SRC_DIR := src
TESTS_DIR := tests
LOGS_DIR := logs
MODELS_DIR := models
DATA_DIR := data
RESULTS_DIR := results

# Default target
.PHONY: help
help: ## Show this help message
	@echo "AI Security Labs - Simple Build System"
	@echo "======================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
.PHONY: setup
setup: clean venv install ## Complete environment setup
	@echo "✅ Environment setup complete!"

.PHONY: venv
venv: ## Create virtual environment
	@echo "📦 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual environment created at $(VENV)"

.PHONY: install
install: $(VENV) ## Install project dependencies
	@echo "📚 Installing dependencies..."
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r requirements.txt
	@echo "✅ Dependencies installed"

.PHONY: install-dev
install-dev: $(VENV) ## Install development dependencies
	$(PIP_VENV) install -r requirements-dev.txt

.PHONY: dirs
dirs: ## Create necessary project directories
	@echo "📁 Creating project directories..."
	@mkdir -p $(LOGS_DIR) $(MODELS_DIR) $(RESULTS_DIR)
	@mkdir -p $(MODELS_DIR)/owasp
	@mkdir -p $(RESULTS_DIR)/owasp/ml01
	@echo "✅ Project directories created"

# Development commands
.PHONY: lint
lint: $(VENV) ## Run code linting
	@echo "🔍 Running linter..."
	$(VENV_BIN)/flake8 $(SRC_DIR) --max-line-length=88 --extend-ignore=E203,W503
	@echo "✅ Linting complete"

.PHONY: test
test: $(VENV) ## Run tests
	@echo "🧪 Running tests..."
	$(VENV_BIN)/pytest $(TESTS_DIR) -v
	@echo "✅ Tests complete"

# OWASP ML01 lab commands
.PHONY: ml01-train
ml01-train: $(VENV) dirs ## Train ML01 model
	@echo "🎯 Training OWASP ML01 model..."
	$(PYTHON_VENV) -m src.owasp.ml01_input_manipulation.train_model
	@echo "✅ OWASP ML01 model training complete"

.PHONY: ml01-attack
ml01-attack: $(VENV) ## Run ML01 FGSM attack
	@echo "⚔️  Running OWASP ML01 FGSM attack..."
	$(PYTHON_VENV) -m src.owasp.ml01_input_manipulation.attack_fgsm
	@echo "✅ OWASP ML01 attack complete"

.PHONY: ml01-defense
ml01-defense: $(VENV) ## Run ML01 defense mechanism
	@echo "🛡️  Running OWASP ML01 defense..."
	$(PYTHON_VENV) -m src.owasp.ml01_input_manipulation.defense_fgsm
	@echo "✅ OWASP ML01 defense complete"

.PHONY: ml01-full
ml01-full: ml01-train ml01-attack ml01-defense ## Run complete ML01 pipeline
	@echo "🚀 OWASP ML01 full pipeline complete!"

# Cleanup commands
.PHONY: clean
clean: ## Clean up temporary files and virtual environment
	@echo "🧹 Cleaning up..."
	rm -rf $(VENV)
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

.PHONY: clean-logs
clean-logs: ## Clean log files
	@echo "🧹 Cleaning logs..."
	rm -rf $(LOGS_DIR)/*.log
	@echo "✅ Logs cleaned"

.PHONY: clean-results
clean-results: ## Clean result files
	@echo "🧹 Cleaning results..."
	rm -rf $(RESULTS_DIR)/*
	@mkdir -p $(RESULTS_DIR)/owasp/ml01
	@echo "✅ Results cleaned"

# Utility commands
.PHONY: logs
logs: ## View recent logs
	@echo "📋 Recent logs:"
	@tail -n 50 $(LOGS_DIR)/ai-security-labs.log 2>/dev/null || echo "No logs found. Run 'make ml01-full' to generate logs."

# Virtual environment check
$(VENV):
	@echo "❌ Virtual environment not found. Run 'make venv' first."
	@exit 1
