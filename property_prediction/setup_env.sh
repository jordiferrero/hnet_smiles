#!/bin/bash
# Setup script for property prediction environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up Property Prediction Environment ==="

# Create virtual environment
if [ ! -d "venv_property" ]; then
    echo "Creating virtual environment..."
    python -m venv venv_property
else
    echo "Virtual environment already exists"
fi

# Activate
source venv_property/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt

# Download datasets if not present
echo "Checking datasets..."
python -c "from scripts.download_datasets import download_all; download_all()" 2>/dev/null || \
    echo "Run 'python scripts/download_datasets.py' to download MoleculeNet datasets"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source venv_property/bin/activate"





