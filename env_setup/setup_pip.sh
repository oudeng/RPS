#!/bin/bash
# Setup environment using pip (with venv)
# Usage: bash setup_pip.sh

set -e

ENV_NAME="rps310"

echo "=== Creating Virtual Environment: ${ENV_NAME} ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Detected Python version: ${PYTHON_VERSION}"

if [[ "${PYTHON_VERSION}" != "3.10" ]]; then
    echo "Warning: Python 3.10 is recommended. Current version: ${PYTHON_VERSION}"
fi

# Remove existing venv if it exists
if [ -d "${ENV_NAME}" ]; then
    echo "Removing existing virtual environment: ${ENV_NAME}"
    rm -rf ${ENV_NAME}
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv ${ENV_NAME}

# Activate and install
echo "Installing packages..."
source ${ENV_NAME}/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
echo "Installing PyTorch with CUDA 11.8..."
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with:"
echo "  source ${ENV_NAME}/bin/activate"