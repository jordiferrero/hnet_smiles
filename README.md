# HNet SMILES Training and Visualization

This repository adapts the HNet architecture for training on SMILES (Simplified Molecular Input Line Entry System) polymer data, focusing on understanding dynamic chunking behavior.

## Overview

The project implements:
- Training pipeline for HNet on SMILES data (~995K polymer entries)
- Incremental training approach (small → medium → large)
- Visualization tools for dynamic chunking (animated GIFs)
- Support for Mac M-chip (MPS) and CUDA devices

## Project Structure

```
LPPXB/
├── setup/                  # Environment setup scripts
│   ├── requirements.txt
│   └── setup_env.sh
├── data/                   # Data loading and analysis
│   ├── analyze_smiles.py
│   └── smiles_dataset.py
├── configs/                # Model configurations
│   ├── hnet_smiles_small.json
│   ├── hnet_smiles_medium.json
│   └── hnet_smiles_large.json
├── visualizations/         # Visualization tools
│   ├── visualize_chunking.py  # Animated GIF generation
│   ├── visualize_stats.py     # Training statistics
│   └── utils.py               # Visualization utilities
├── train_smiles.py        # Main training script
├── generate_smiles.py      # SMILES generation script
└── datasets/
    └── PI1M/
        └── PI1M_v2.csv    # SMILES dataset (~995K entries)
```

## Setup

### 1. Create Virtual Environment

```bash
cd setup
./setup_env.sh
source ../venv/bin/activate
```

The setup script automatically detects your platform (Mac M-chip, CUDA, or CPU) and installs the appropriate dependencies.

### 2. Verify Installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'Device: {torch.cuda.is_available() or torch.backends.mps.is_available()}')"
```

## Usage

### Data Analysis

Analyze the SMILES dataset to understand length distribution:

```bash
python data/analyze_smiles.py --csv-path datasets/PI1M/PI1M_v2.csv --plot
```

### Training

Train the model in phases:

**Phase 1 (Small - 1K entries):**
```bash
python train_smiles.py \
    --config configs/hnet_smiles_small.json \
    --phase small \
    --max-samples 1000 \
    --batch-size 8 \
    --epochs 5 \
    --output-dir checkpoints
```

**Phase 2 (Medium - 10K entries):**
```bash
python train_smiles.py \
    --config configs/hnet_smiles_medium.json \
    --phase medium \
    --max-samples 10000 \
    --batch-size 16 \
    --epochs 10 \
    --output-dir checkpoints
```

**Phase 3 (Large - Full dataset):**
```bash
python train_smiles.py \
    --config configs/hnet_smiles_large.json \
    --phase large \
    --batch-size 32 \
    --epochs 20 \
    --output-dir checkpoints
```

### Visualization

Create animated GIFs showing dynamic chunking:

```bash
python visualizations/visualize_chunking.py \
    --checkpoint checkpoints/checkpoint_phase_small_epoch_5.pt \
    --config configs/hnet_smiles_small.json \
    --text "*CCC[Fe]CCCC(=O)OCCCCOCCCNCC(*)=O" \
    --output visualizations/output/chunking_example.gif
```

Visualize multiple SMILES strings:

```bash
python visualizations/visualize_chunking.py \
    --checkpoint checkpoints/checkpoint_phase_small_epoch_5.pt \
    --config configs/hnet_smiles_small.json \
    --text-file datasets/PI1M/PI1M_v2.csv \
    --num-samples 10 \
    --output visualizations/output/
```

Plot training statistics:

```bash
python visualizations/visualize_stats.py \
    --checkpoint-dir checkpoints \
    --checkpoint checkpoints/checkpoint_phase_small_epoch_5.pt \
    --config configs/hnet_smiles_small.json \
    --output-dir visualizations/stats
```

### Generation

Generate SMILES strings from a trained model:

```bash
python generate_smiles.py \
    --checkpoint checkpoints/checkpoint_phase_small_epoch_5.pt \
    --config configs/hnet_smiles_small.json \
    --prompt "*" \
    --max-tokens 512 \
    --temperature 1.0
```

## Key Features

### Dynamic Chunking Visualization

The visualization tools create animated GIFs similar to the original HNet paper, showing:
- Hex encoding of SMILES strings
- Progressive chunking as tokens are processed
- Green squares marking chunk boundaries
- Blue U-shapes showing chunk spans
- Frame-by-frame evolution of chunking

### Incremental Training

The training pipeline supports three phases:
1. **Small**: Quick validation on 1K entries
2. **Medium**: Hyperparameter tuning on 10K-100K entries
3. **Large**: Full training on ~995K entries

### Platform Support

- **Mac M-chip**: Uses PyTorch MPS backend (Metal Performance Shaders)
- **CUDA**: Full CUDA support for GPU acceleration
- **CPU**: Fallback for testing without GPU

## Configuration

Model configurations are in JSON format, adapting the original HNet configs:
- `hnet_smiles_small.json`: Minimal config for testing (512/768 dims)
- `hnet_smiles_medium.json`: Medium-scale config (768/1024 dims)
- `hnet_smiles_large.json`: Large-scale config (1024/1536 dims)

All configs use:
- `vocab_size: 256` (ByteTokenizer compatible)
- 1-stage architecture: `["m4", ["T22"], "m4"]`
- Appropriate SSM and attention configurations

## Dataset

The PI1M dataset contains ~995K SMILES strings representing polymers. Each entry is a SMILES string in the format:
```
SMILES,SA Score
*CCC[Fe]CCCC(=O)OCCCCOCCCNCC(*)=O,4.174851129781874
...
```

The training pipeline extracts SMILES from the first column and optionally concatenates multiple SMILES for longer sequences.

## Notes

- SMILES strings are relatively short (typically 10-50 characters), so concatenation may be beneficial for chunking analysis
- The model uses byte-level tokenization (ByteTokenizer) compatible with SMILES
- Load balancing loss is weighted at 0.01 relative to cross-entropy loss
- Training uses mixed precision (bfloat16) on GPU/MPS devices

## Troubleshooting

### flash_attn not available on Mac
This is expected. The model will use alternative attention mechanisms.

### Out of memory errors
Reduce batch size or use gradient accumulation:
```bash
python train_smiles.py ... --batch-size 4 --gradient-accumulation 2
```

### Import errors
Make sure the virtual environment is activated and hnet package is installed:
```bash
source venv/bin/activate
cd original_resources/hnet-github-repo
pip install -e .
```

## References

- Original HNet paper: [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955)
- Original HNet repository: https://github.com/goombalab/hnet
- PI1M dataset: https://github.com/RUIMINMA1996/PI1M

