#!/bin/bash

# Remove existing venv if it exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Create new venv with Python 3.7
echo "Creating new virtual environment with Python 3.7..."
uv venv --python 3.7

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install package in editable mode with dev dependencies
echo "Installing package and dependencies..."
uv pip install -e ".[dev]"

# Setup test data
echo "Setting up test data..."
python setup_tests.py

# Run tests
echo "Running tests..."
pytest -v tests/
