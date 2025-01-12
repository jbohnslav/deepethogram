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

# Install requirements first
echo "Installing requirements..."
uv pip install -r requirements.txt

# Install pytest
echo "Installing pytest..."
uv pip install pytest pytest-cov

# Install package in editable mode
echo "Installing package in editable mode..."
uv pip install -e .

# Run tests
echo "Running tests..."
pytest -v tests/
