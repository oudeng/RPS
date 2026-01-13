# Environment Setup Guide

This guide provides instructions for setting up the Python environment required to reproduce the experiments.

## Prerequisites

- **Operating System**: Linux (Ubuntu 18.04+), macOS, or Windows with WSL2
- **Python**: 3.10.x
- **GPU** (optional but recommended): NVIDIA GPU with CUDA 11.8 support

## Quick Start

We provide two installation methods. Choose the one that best fits your setup.

### Option 1: Conda (Recommended)

```bash
bash setup_conda.sh
conda activate rps310
```

### Option 2: pip + venv

```bash
bash setup_pip.sh
source rps310/bin/activate
```

## Installation Details

### Method 1: Conda Environment

This method is recommended for most users as it handles system-level dependencies automatically.

**Step 1**: Ensure [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed.

**Step 2**: Run the setup script:

```bash
bash setup_conda.sh
```

**Step 3**: Activate the environment:

```bash
conda activate rps310
```

Alternatively, you can create the environment manually:

```bash
conda env create -f environment.yml
conda activate rps310
```

### Method 2: pip with Virtual Environment

This method is suitable for users who prefer a lightweight setup without Conda.

**Step 1**: Ensure Python 3.10 is installed:

```bash
python3 --version  # Should output Python 3.10.x
```

**Step 2**: Run the setup script:

```bash
bash setup_pip.sh
```

**Step 3**: Activate the environment:

```bash
source rps310/bin/activate
```

Alternatively, you can set up manually:

```bash
python3 -m venv rps310
source rps310/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Verifying the Installation

After activation, verify that all packages are installed correctly:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import numpy, pandas, scipy, sklearn; print('All dependencies imported successfully')"
```

## Troubleshooting

### CUDA Not Available

If `torch.cuda.is_available()` returns `False`:

1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Ensure CUDA 11.8 is compatible with your GPU
3. For CPU-only usage, install PyTorch without CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

### Conda Environment Creation Fails

If you encounter dependency conflicts:

```bash
conda env create -f environment.yml --force
```

### Permission Denied

Make the scripts executable:

```bash
chmod +x setup_conda.sh setup_pip.sh
```

## File Structure

```
.
├── environment.yml      # Conda environment specification
├── requirements.txt     # pip requirements
├── setup_conda.sh       # Conda installation script
├── setup_pip.sh         # pip installation script
└── README_SETUP.md      # This file
```

## Dependencies

| Package      | Version   | Description                          |
|--------------|-----------|--------------------------------------|
| Python       | 3.10      | Programming language                 |
| PyTorch      | ≥1.12     | Deep learning framework              |
| NumPy        | ≥1.23     | Numerical computing                  |
| Pandas       | ≥1.5      | Data manipulation                    |
| SciPy        | ≥1.11     | Scientific computing                 |
| Scikit-learn | ≥1.2      | Machine learning utilities           |
| XGBoost      | ≥1.7      | Gradient boosting                    |
| Matplotlib   | ≥3.7      | Visualization                        |
| Seaborn      | ≥0.12     | Statistical visualization            |

## License

Please refer to the [LICENSE](LICENSE) file in the root directory.