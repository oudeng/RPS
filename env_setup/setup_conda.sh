#!/bin/bash
# Setup environment using Conda
# Usage: bash setup_conda.sh

set -e

ENV_NAME="rps310"

echo "=== Creating Conda Environment: ${ENV_NAME} ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove -n ${ENV_NAME} -y
fi

# Create environment from yml file
echo "Creating environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with:"
echo "  conda activate ${ENV_NAME}"