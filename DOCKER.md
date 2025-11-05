# Docker Setup for HNet SMILES Training

This guide explains how to run HNet SMILES training in a Docker container on an EC2 instance with GPU.

## Prerequisites

- Docker installed on EC2 instance
- NVIDIA Docker runtime (nvidia-docker2) installed
- CUDA 12.1+ compatible GPU

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t hnet-smiles:latest .
```

This will:
- Install all system dependencies
- Install PyTorch with CUDA 12.1 support
- Build and install mamba_ssm, causal-conv1d, and flash-attn
- Install the HNet package
- Verify the installation

### 2. Run the Container

#### Option A: Using Docker Compose (Recommended)

```bash
docker-compose up -d
docker-compose exec hnet-smiles bash
```

#### Option B: Using Docker Run

```bash
docker run --gpus all -it \
  -v $(pwd)/datasets:/workspace/datasets:ro \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/visualizations:/workspace/visualizations \
  -v $(pwd)/configs:/workspace/configs:ro \
  --name hnet-smiles \
  hnet-smiles:latest
```

### 3. Verify Installation Inside Container

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python3 -c "import hnet; print('HNet package installed')"
python3 -c "import mamba_ssm; print('mamba_ssm installed')"
python3 -c "import flash_attn; print('flash_attn installed')" || echo "flash_attn not available (may be OK)"
```

### 4. Run Training

Inside the container:

```bash
# Analyze dataset (optional)
python3 data/analyze_smiles.py --csv-path datasets/PI1M/PI1M_v2.csv --plot --output-dir visualizations

# Train small phase
python3 train_smiles.py \
    --config configs/hnet_smiles_small.json \
    --phase small \
    --max-samples 1000 \
    --batch-size 8 \
    --epochs 5 \
    --output-dir checkpoints

# Generate SMILES
python3 generate_smiles.py \
    --checkpoint checkpoints/checkpoint_phase_small_epoch_5.pt \
    --config configs/hnet_smiles_small.json \
    --prompt "*" \
    --max-tokens 512
```

## Docker Image Details

### Base Image
- `nvidia/cuda:12.1.0-devel-ubuntu22.04`
- Provides CUDA 12.1 and development tools (nvcc, etc.)

### Dependencies Installed

1. **System Dependencies:**
   - Python 3.11
   - Build tools (gcc, make, cmake, ninja)
   - Git

2. **PyTorch:**
   - PyTorch with CUDA 12.1 support
   - Installed from PyTorch official repository

3. **HNet Dependencies:**
   - `triton>=3.2.0` - Required for HNet operations
   - `mamba_ssm` - Built from source from GitHub
   - `causal-conv1d` - Built from source from GitHub
   - `flash-attn==2.8.0.post2` - Optional but recommended for performance

4. **Project Dependencies:**
   - All packages from `setup/requirements.txt`
   - HNet package installed in editable mode

## Volume Mounts

The Docker setup mounts the following directories:

- `datasets/` - Read-only mount for training data
- `checkpoints/` - Read-write mount for model checkpoints
- `visualizations/` - Read-write mount for visualization outputs
- `configs/` - Read-only mount for configuration files

This allows you to:
- Keep data outside the container
- Persist checkpoints across container restarts
- Access visualization outputs from the host

## Troubleshooting

### CUDA Not Available

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory During Build

If the build fails due to memory issues, try building with more memory:

```bash
docker build --memory=8g -t hnet-smiles:latest .
```

### Flash Attention Build Fails

Flash attention may fail to build on some systems. The Dockerfile will continue without it, and the model will use alternative attention mechanisms. This is acceptable for training.

### Rebuild After Code Changes

If you modify the code, rebuild the image:

```bash
docker build -t hnet-smiles:latest .
```

Or if using docker-compose:

```bash
docker-compose build
```

## Running on EC2

### EC2 Instance Setup

1. Launch an EC2 instance with:
   - GPU instance type (e.g., g4dn.xlarge, p3.2xlarge)
   - Ubuntu 22.04 LTS
   - NVIDIA GPU drivers installed

2. Install Docker and NVIDIA Docker runtime:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

3. Clone repository and build:

```bash
git clone <your-repo>
cd hnet_smiles
docker build -t hnet-smiles:latest .
```

4. Run training:

```bash
docker run --gpus all -d \
  -v $(pwd)/datasets:/workspace/datasets:ro \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/visualizations:/workspace/visualizations \
  --name hnet-smiles \
  hnet-smiles:latest \
  python3 train_smiles.py --config configs/hnet_smiles_small.json --phase small --max-samples 1000 --batch-size 8 --epochs 5 --output-dir checkpoints
```

## Performance Tips

1. **Use flash-attn**: If flash-attn builds successfully, it will significantly improve training speed.

2. **Batch Size**: Adjust batch size based on GPU memory:
   ```bash
   --batch-size 16  # For 16GB GPU
   --batch-size 32  # For 24GB+ GPU
   ```

3. **Gradient Accumulation**: Use gradient accumulation for larger effective batch sizes:
   ```bash
   --batch-size 8 --gradient-accumulation 4  # Effective batch size: 32
   ```

4. **Mixed Precision**: Enabled by default (bfloat16) for better performance and memory usage.

