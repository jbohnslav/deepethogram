#!/bin/bash

# Remove existing venv if it exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Create new venv with Python 3.11
echo "Creating new virtual environment with Python 3.11..."
uv venv --python 3.11

# Install project and dev dependencies from the lockfile
echo "Installing package and dependencies..."
uv sync --dev

# Setup test data
echo "Setting up test data..."
uv run python setup_tests.py

# Run tests
echo "Running tests..."
uv run pytest -v tests/
