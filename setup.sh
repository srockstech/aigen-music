#!/bin/bash

# Exit on error
set -e

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install it first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Remove existing virtual environment if it exists
poetry env remove --all

# Clear poetry cache
poetry cache clear . --all

echo "Installing dependencies..."
poetry install

# Install spacy model
poetry run python -m spacy download en_core_web_sm

echo "Setup completed! You can now activate the environment with 'poetry shell'" 