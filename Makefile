.PHONY: install lint fix format test clean run train

# Install the package and dev dependencies in editable mode
install:
	pip install -e ".[dev]"

# Check for linting errors without modifying files
lint:
	ruff check src/ tests/

# Auto-fix formatting and linting issues in place
fix:
	ruff format . && ruff check --fix .

# Format source files without touching imports or lint rules
format:
	ruff format src/ tests/

# Run the full test suite with verbose output
test:
	pytest tests/ -v

# Remove Python bytecode and __pycache__ directories
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	uvicorn src.api.main:app --reload --port 8000

# Train MLP and baseline models.
# Requires the following to be running in another terminal:
#   mlflow ui --port 5001
train:
	python -m src.models.mlp_baselines