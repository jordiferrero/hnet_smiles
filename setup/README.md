# Environment Setup

This directory contains scripts to set up the Python environment for HNet SMILES training.

## Quick Start

```bash
cd setup
./setup_env.sh
source ../venv/bin/activate
```

## Platform Support

### Mac M-chip (Apple Silicon)
- Uses PyTorch MPS backend (Metal Performance Shaders)
- `flash_attn` may not be available - the model will use fallback attention mechanisms
- All other dependencies should work normally

### Linux with CUDA
- Uses PyTorch CUDA backend
- Full support for all dependencies including `flash_attn`
- Requires CUDA 12.1+ and compatible GPU

### CPU-only
- Works on any platform
- Slower but functional for testing
- No GPU acceleration

## Manual Installation

If the script fails, you can install manually:

1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install PyTorch (choose appropriate version):
   - Mac M-chip: `pip install torch torchvision torchaudio`
   - CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
   - CPU: `pip install torch torchvision torchaudio`

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Install git dependencies:
   ```bash
   pip install git+https://github.com/state-spaces/mamba.git@a6a1dae6efbf804c9944a0c2282b437deb4886d8
   pip install git+https://github.com/Dao-AILab/causal-conv1d.git@e940ead2fd962c56854455017541384909ca669f
   ```

5. Install hnet package:
   ```bash
   cd ../original_resources/hnet-github-repo
   pip install -e .
   ```

## Troubleshooting

### flash_attn fails on Mac
This is expected. The model will use alternative attention mechanisms. You can continue without it.

### mamba_ssm compilation fails
Make sure you have a C++ compiler installed:
- Mac: `xcode-select --install`
- Linux: `sudo apt-get install build-essential`

### CUDA not detected
Check CUDA installation: `nvidia-smi`. If not available, the script will default to CPU mode.

