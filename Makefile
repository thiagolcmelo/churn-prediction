.PHONY: install lint fix format test clean

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
