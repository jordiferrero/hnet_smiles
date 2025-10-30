#!/bin/bash
# Setup script for HNet SMILES training environment
# Supports Mac M-chip (MPS) and CUDA devices

set -e  # Exit on error

echo "=========================================="
echo "HNet SMILES Environment Setup"
echo "=========================================="

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"

# Check if we're on Mac
if [[ "$OS" == "Darwin" ]]; then
    echo "Mac detected - will use MPS backend"
    USE_MPS=true
    USE_CUDA=false
elif [[ "$OS" == "Linux" ]]; then
    echo "Linux detected - checking for CUDA"
    USE_MPS=false
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA detected"
        USE_CUDA=true
    else
        echo "No CUDA detected - will use CPU"
        USE_CUDA=false
    fi
else
    echo "Unknown OS - defaulting to CPU"
    USE_MPS=false
    USE_CUDA=false
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch based on platform
echo ""
echo "Installing PyTorch..."
if [[ "$USE_MPS" == true ]]; then
    echo "Installing PyTorch with MPS support (Mac M-chip)..."
    pip install torch torchvision torchaudio
elif [[ "$USE_CUDA" == true ]]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio
fi

# Install base requirements
echo ""
echo "Installing base requirements..."
pip install -r requirements.txt

# Install git dependencies
echo ""
echo "Installing git dependencies..."

# Install mamba_ssm
echo "Installing mamba_ssm..."
pip install git+https://github.com/state-spaces/mamba.git@a6a1dae6efbf804c9944a0c2282b437deb4886d8

# Install causal_conv1d
echo "Installing causal_conv1d..."
pip install git+https://github.com/Dao-AILab/causal-conv1d.git@e940ead2fd962c56854455017541384909ca669f

# Install flash_attn (may fail on Mac M-chip, but try anyway)
echo "Installing flash_attn..."
if [[ "$USE_MPS" == true ]]; then
    echo "WARNING: flash_attn may not work on Mac M-chip. Skipping..."
    echo "If needed, you may need to compile from source or use CPU fallback."
else
    pip install flash-attn==2.8.0.post2 --no-build-isolation || {
        echo "WARNING: flash_attn installation failed. Continuing without it..."
        echo "The model may fall back to alternative attention mechanisms."
    }
fi

# Install hnet package
echo ""
echo "Installing hnet package..."
cd ../original_resources/hnet-github-repo
pip install -e .
cd ../../setup

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 << EOF
import sys
import torch

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check device availability
if torch.backends.mps.is_available():
    print("✓ MPS (Metal) backend available")
elif torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ Using CPU (no GPU acceleration)")

# Test imports
try:
    import hnet
    print("✓ hnet package installed")
except ImportError as e:
    print(f"✗ hnet import failed: {e}")

try:
    import mamba_ssm
    print("✓ mamba_ssm installed")
except ImportError as e:
    print(f"✗ mamba_ssm import failed: {e}")

try:
    import flash_attn
    print("✓ flash_attn installed")
except ImportError:
    print("⚠ flash_attn not available (may be OK on Mac M-chip)")

print("\nSetup complete!")
print("\nTo activate the environment, run:")
print("  source venv/bin/activate")
EOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run data analysis: python data/analyze_smiles.py"
echo "3. Start training: python train_smiles.py --phase small"
echo ""

