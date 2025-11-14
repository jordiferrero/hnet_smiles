# HNet SMILES Training

Training pipeline for HNet (Hierarchical Network) on SMILES (Simplified Molecular Input Line Entry System) polymer and molecular data, with automatic visualization of dynamic chunking behavior during training.

![Training Evolution](static/training_evolution.gif)

*Dynamic chunking behavior evolution during HNet training on SMILES data*

## Quick Start

**Three simple steps:**

1. **Edit config file**: `configs/hnet_smiles_large.json` (or `hnet_smiles_small.json` for testing)
2. **Activate environment**: `cd /home/ec2-user/hnet_smiles && source /opt/pytorch/bin/activate`
3. **Run training**: `python train_smiles.py --config configs/hnet_smiles_large.json`

Training automatically generates checkpoints and visualizations upon completion!

## Overview

This project implements a complete training pipeline for HNet on SMILES data with:

- **Config-based training**: All parameters in JSON configs - minimal command-line arguments needed
- **Intelligent checkpointing**: 
  - Boundary predictions saved to compressed pickle files at regular byte intervals
  - Model checkpoints only saved when training loss improves (saves disk space)
- **Automatic visualization**: Training evolution GIFs generated automatically at the end
- **Hierarchical learning rate modulation**: Different learning rates for encoder/decoder vs. main network
- **Graceful shutdown**: Signal handling for SSH disconnections and clean checkpoint saving
- **Resume capability**: Resume training from any checkpoint

## Installation

### On EC2 Instance (Pre-configured)

If you're working on the EC2 instance with pre-configured PyTorch environment:

```bash
source /opt/pytorch/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Local Setup (Mac/Linux)

For local installation from scratch:

1. **Create Virtual Environment**:
   ```bash
   cd setup
   ./setup_env.sh
   source ../venv/bin/activate
   ```

2. **Verify Installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'Device: {torch.cuda.is_available() or torch.backends.mps.is_available()}')"
   ```

The setup script automatically detects your platform (Mac M-chip, CUDA, or CPU) and installs the appropriate dependencies. See `setup/README.md` for detailed instructions and troubleshooting.

## Project Structure

```
hnet_smiles/
├── configs/                # Model configurations (edit these!)
│   ├── hnet_smiles_small.json  # Small config for testing
│   └── hnet_smiles_large.json  # Large config for full training
├── data/                   # Data loading modules
│   ├── analyze_smiles.py   # SMILES data analysis utilities
│   └── smiles_dataset.py   # PyTorch dataset implementation
├── datasets/               # Actual data files
│   ├── PI1M/
│   │   └── PI1M_v2.csv     # Polymer SMILES dataset (~995K entries)
│   └── moses/
│       └── smiles-molecules-moses_all.csv  # Molecular SMILES dataset
├── visualizations/         # Visualization tools
│   ├── visualize_training_evolution.py  # Training evolution GIFs
│   ├── visualize_stats.py              # Training statistics plots
│   └── utils.py                        # Visualization utilities
├── analysis/               # Analysis notebooks and reports
│   ├── notebooks/          # Jupyter notebooks for analysis
│   ├── data/              # Analysis results and statistics
│   ├── figures/           # Generated figures
│   └── FINAL_REPORT.md    # Comprehensive analysis report
├── static/                 # Static assets
│   └── training_evolution.gif  # Example training visualization
├── train_smiles.py        # Main training script
├── generate_smiles.py     # SMILES generation script
└── checkpoints/           # Training outputs (created automatically)
    └── run_{phase}_{timestamp}/
        ├── checkpoints/   # Model checkpoints
        │   ├── checkpoint_epoch_*.pt  # Epoch checkpoints
        │   └── checkpoint_bytes_best.pt  # Best checkpoint
        ├── visualizations/
        │   ├── predictions/  # Compressed pickle files with boundary predictions
        │   │   └── predictions_bytes_*.pkl.gz
        │   ├── training_evolution_batch_1.gif  # Batch 1 visualization
        │   ├── training_evolution_batch_2.gif  # Batch 2 visualization
        │   └── ...                             # Additional batch visualizations
        ├── metadata.json      # Training metadata and history
        ├── test_smiles.txt    # Test SMILES used for visualization
        └── run_purpose.txt    # Purpose/notes for this run
```

## Usage

### Basic Training

1. **Edit the config file** (`configs/hnet_smiles_large.json` or `hnet_smiles_small.json`):
   - Adjust model architecture, training parameters, dataset path, etc.
   - All training parameters are in the `training_config` section

2. **Activate the virtual environment**:
   ```bash
   cd /home/ec2-user/hnet_smiles
   source /opt/pytorch/bin/activate
   ```

3. **Run training**:
   ```bash
   python train_smiles.py --config configs/hnet_smiles_large.json
   ```

### Running Training in Background (SSH Disconnection Safe)

For long-running training sessions, use `nohup` to prevent termination when SSH disconnects:

```bash
nohup python train_smiles.py --config configs/hnet_smiles_large.json > training.log 2>&1 &
```

**Monitor progress** (when reconnected):
```bash
# View live log output
tail -f training.log

# Check if training is still running
ps aux | grep train_smiles.py

# View latest checkpoint directory
ls -lth checkpoints/ | head -5

# View training metadata
cat checkpoints/run_*/metadata.json | tail -20
```

**Stop training** (if needed):
```bash
# Find the process ID
ps aux | grep train_smiles.py

# Kill the process (replace PID with actual process ID)
kill PID
```

The training script includes signal handlers for graceful shutdown - it will save a checkpoint before exiting when receiving SIGTERM or SIGINT.

### Resuming Training

To resume from a checkpoint:

```bash
python train_smiles.py --config configs/hnet_smiles_large.json \
    --resume checkpoints/run_large_20251110_164118/checkpoints/checkpoint_epoch_3.pt
```

The script will automatically:
- Restore model and optimizer state
- Resume from the correct epoch
- Continue tracking cumulative training bytes
- Preserve best loss tracking

### Command-Line Arguments

While all parameters are in the config file, you can override them via command-line:

```bash
python train_smiles.py --config configs/hnet_smiles_large.json \
    --batch-size 32 \
    --epochs 10 \
    --lr 0.0002 \
    --checkpoint-bytes 2000000
```

See `python train_smiles.py --help` for all available arguments.

## Configuration

### Model Architecture

Model configurations are in JSON format with two main sections:

#### Architecture Parameters

- `arch_layout`: Network architecture layout (e.g., `["m4", ["T22"], "m4"]` for 1-stage)
- `d_model`: Model dimensions for each stage (e.g., `[1024, 1536]`)
- `d_intermediate`: Intermediate dimensions for MLP layers (e.g., `[0, 4096]`)
- `vocab_size`: Vocabulary size (256 for ByteTokenizer)
- `ssm_cfg`: State Space Model configuration
  - `chunk_size`: Chunk size for SSM processing
  - `d_conv`: Convolution dimension
  - `d_state`: State dimension
  - `expand`: Expansion factor
- `attn_cfg`: Attention configuration
  - `num_heads`: Number of attention heads per stage
  - `rotary_emb_dim`: Rotary embedding dimension per stage
  - `window_size`: Attention window size per stage (-1 for full attention)

#### Training Configuration

All training parameters are in the `training_config` section:

- `data`: Path to SMILES CSV file
- `phase`: Training phase identifier (small/medium/large)
- `max_samples`: Maximum number of samples (null = all)
- `batch_size`: Batch size for training
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `weight_decay`: Weight decay for optimizer
- `gradient_accumulation`: Gradient accumulation steps
- `concatenate`: Whether to concatenate multiple SMILES
- `num_concatenate`: Number of SMILES to concatenate (if concatenate=true)
- `concatenate_separator`: Separator between concatenated SMILES
- `checkpoint_bytes`: Save predictions every N training bytes
- `num_test_samples`: Total number of test samples to evaluate
- `num_visualize`: Number of samples to visualize in GIF
- `skip_visualization`: Skip visualization generation (default: false)
- `output_dir`: Directory for checkpoints and outputs
- `no_amp`: Disable automatic mixed precision (default: false)

### Example Config Structure

```json
{
  "arch_layout": ["m4", ["T22"], "m4"],
  "d_model": [1024, 1536],
  "d_intermediate": [0, 4096],
  "vocab_size": 256,
  "ssm_cfg": {
    "chunk_size": 256,
    "d_conv": 4,
    "d_state": 128,
    "expand": 2
  },
  "attn_cfg": {
    "num_heads": [16, 16],
    "rotary_emb_dim": [32, 48],
    "window_size": [1023, -1]
  },
  "tie_embeddings": false,
  "training_config": {
    "data": "datasets/PI1M/PI1M_v2.csv",
    "phase": "large",
    "max_samples": null,
    "batch_size": 16,
    "epochs": 5,
    "lr": 0.0001,
    "weight_decay": 0.1,
    "gradient_accumulation": 8,
    "concatenate": true,
    "num_concatenate": 10,
    "concatenate_separator": " ",
    "checkpoint_bytes": 1000000,
    "num_test_samples": 50,
    "num_visualize": 15,
    "skip_visualization": false,
    "output_dir": "checkpoints",
    "no_amp": false
  }
}
```

### Pre-configured Models

- **`hnet_smiles_small.json`**: Small config for testing (512/768 dims, 2 epochs, 5000 samples)
- **`hnet_smiles_large.json`**: Large-scale config (1024/1536 dims, configurable epochs, full dataset)

## Training Output

After training, you'll find in `checkpoints/run_{phase}_{timestamp}/`:

### Checkpoints

- **Epoch checkpoints**: `checkpoint_epoch_{N}.pt` - Saved at the end of each epoch
- **Byte-based checkpoints**: `checkpoint_bytes_{N}.pt` - Saved only when training loss improves
  - Checkpoints are saved at regular byte intervals (every `checkpoint_bytes`)
  - But only when the current loss is better than the previous best loss
  - This saves disk space while preserving the best models

### Predictions

- **`visualizations/predictions/`**: Compressed pickle files with boundary predictions
  - `predictions_bytes_1,000,000.pkl.gz`, `predictions_bytes_2,000,000.pkl.gz`, etc.
  - One file per checkpoint interval (saved regardless of loss improvement)
  - Contains boundary predictions for all test samples
  - Much smaller than JSON (typically 5-10x compression)

### Visualizations

- **`visualizations/training_evolution_batch_N.gif`**: Animated GIFs showing chunking evolution
  - Multiple GIFs generated, one per batch of test samples
  - Shows how boundary predictions evolve across training checkpoints
  - Generated automatically at the end of training
  - Uses saved prediction files (no model reloading needed)
  - Each GIF displays the tokenization behavior for a subset of test SMILES

### Metadata

- **`metadata.json`**: Complete training metadata including:
  - Model configuration
  - Training arguments
  - Dataset information
  - Training history (loss, bytes, checkpoints)
  - Model info (parameters, device, dtype)

- **`test_smiles.txt`**: Test SMILES used for visualization (one per line)
- **`run_purpose.txt`**: Optional notes about the purpose of this training run

## Analysis Framework

The `analysis/` folder contains a comprehensive framework for analyzing HNet's tokenization behavior:

### Analysis Components

- **Jupyter Notebooks** (`notebooks/`):
  - `01_data_generation.ipynb`: Generate tokenization data from trained models
  - `02_dataset_nature_analysis.ipynb`: Compare PI1M (polymers) vs MOSES (molecules)
  - `03_concatenation_effect.ipynb`: Analyze impact of concatenation strategy
  - `04_training_amount_analysis.ipynb`: Study effect of training epochs
  - `05_benchmark_comparison.ipynb`: Compare HNet with SmilesPE baseline

- **Analysis Results** (`data/`):
  - Tokenization statistics for all trained models
  - Comparison summaries (CSV format)
  - Token frequency distributions and statistics

- **Reports**:
  - `FINAL_REPORT.md`: Comprehensive analysis report with key findings
  - Detailed comparison of HNet's learned tokenization vs. traditional methods

This analysis framework enables deep investigation into how HNet learns to tokenize chemical SMILES strings dynamically.

## Key Features

### Config-Based Training

All training parameters are specified in JSON config files. Simply edit the config and run training - no command-line arguments needed (though they can override config values).

### Intelligent Checkpointing

The training script uses a two-tier checkpointing strategy:

1. **Boundary predictions**: Always saved at every `checkpoint_bytes` interval
   - Saved to compressed pickle files (`.pkl.gz`)
   - Used for visualization (no model reloading needed)
   - Much smaller than full model checkpoints

2. **Model checkpoints**: Only saved when training loss improves
   - Saves disk space by not storing every checkpoint
   - Preserves the best models at each byte interval
   - Includes full model state, optimizer state, and metadata

### Hierarchical Learning Rate Modulation

The model applies different learning rates to different parts of the network:

- **Encoder/Decoder stages**: Higher learning rate (3.0x base LR)
- **Main network stages**: Lower learning rate (0.9x base LR)
- **Multi-stage networks**: Interpolated learning rates

This follows the HNet paper's recommendation for hierarchical training.

### Automatic Visualization

Training automatically generates visualization GIFs at the end:

- Boundary predictions are saved during training at every `checkpoint_bytes` interval
- Visualizations are generated from saved predictions (no model reloading needed)
- Shows the evolution of chunking behavior across training steps
- Similar to the original HNet paper visualizations

### Load Balancing Loss

The training includes a load balancing loss term (weighted at 0.01) to encourage uniform token selection across the routing module. This helps maintain balanced chunking behavior.

### Mixed Precision Training

Training uses automatic mixed precision (bfloat16) on GPU/MPS devices by default, which:
- Reduces memory usage
- Speeds up training
- Maintains training stability

Disable with `--no-amp` or `"no_amp": true` in config.

### Graceful Shutdown

The training script handles signals gracefully:

- **SIGHUP**: Ignored (continues training during SSH disconnection)
- **SIGTERM/SIGINT**: Saves checkpoint and exits gracefully

For full SSH disconnection protection, use `nohup`, `screen`, `tmux`, or a systemd service.

## Manual Visualization

If you want to regenerate visualizations from saved predictions:

```bash
python visualizations/visualize_training_evolution.py \
    --run-dir checkpoints/run_large_20251110_164118 \
    --output checkpoints/run_large_20251110_164118/visualizations/
```

This will generate multiple GIF files (`training_evolution_batch_1.gif`, `training_evolution_batch_2.gif`, etc.), one per batch of test samples.

Plot training statistics:

```bash
python visualizations/visualize_stats.py \
    --run-dir checkpoints/run_large_20251110_164118 \
    --output-dir checkpoints/run_large_20251110_164118/visualizations/stats
```

## Generation

Generate SMILES strings from a trained model:

```bash
python generate_smiles.py \
    --checkpoint checkpoints/run_large_20251110_164118/checkpoints/checkpoint_epoch_5.pt \
    --config configs/hnet_smiles_large.json \
    --prompt "*" \
    --max-tokens 512 \
    --temperature 1.0
```

## Datasets

The project supports two SMILES datasets:

### PI1M Dataset
- **Path**: `datasets/PI1M/PI1M_v2.csv`
- **Size**: ~995K SMILES strings representing polymers
- **Format**: CSV with SMILES and SA Score columns
- **Example**: `*CCC[Fe]CCCC(=O)OCCCCOCCCNCC(*)=O,4.174851129781874`

### MOSES Dataset
- **Path**: `datasets/moses/smiles-molecules-moses_all.csv`
- **Size**: Molecular SMILES strings
- **Format**: CSV with SMILES strings

### Training Pipeline
The training pipeline:
- Extracts SMILES from the first column (or the appropriate column)
- Optionally concatenates multiple SMILES for longer sequences (10 by default)
- Uses byte-level tokenization (ByteTokenizer) with 256 vocab size
- Splits into train/test sets with a fixed random seed (42) for reproducibility
- Supports both polymer (PI1M) and molecular (MOSES) SMILES

## Platform Support

- **CUDA**: Full CUDA support for GPU acceleration (EC2 instances)
- **Mac M-chip**: Uses PyTorch MPS backend (Metal Performance Shaders)
- **CPU**: Fallback for testing without GPU

## Troubleshooting

### Out of Memory Errors

Reduce batch size or use gradient accumulation:

```bash
python train_smiles.py --config configs/hnet_smiles_large.json \
    --batch-size 32 \
    --gradient-accumulation 16
```

Or edit the config file to adjust these parameters.

### Import Errors

Make sure the virtual environment is activated:

```bash
cd /home/ec2-user/hnet_smiles
source /opt/pytorch/bin/activate
```

### Disk Space Issues

Training saves predictions at byte intervals. If disk space is limited:

- Increase `checkpoint_bytes` in config to save less frequently
- Set `skip_visualization: true` to skip prediction saving during training
- Clean up old checkpoint directories
- Note: Model checkpoints are only saved when loss improves, so they're already space-efficient

### flash_attn not available on Mac

This is expected. The model will use alternative attention mechanisms.

### Training Stops When SSH Disconnects

Use `nohup` or a terminal multiplexer:

```bash
# Using nohup
nohup python train_smiles.py --config configs/hnet_smiles_large.json > training.log 2>&1 &

# Using screen
screen -S training
python train_smiles.py --config configs/hnet_smiles_large.json
# Detach with Ctrl+A D

# Using tmux
tmux new -s training
python train_smiles.py --config configs/hnet_smiles_large.json
# Detach with Ctrl+B D
```

## References

- **Original HNet paper**: [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955)
- **Original HNet repository**: https://github.com/goombalab/hnet
- **PI1M dataset**: https://github.com/RUIMINMA1996/PI1M (Polymer SMILES)
- **MOSES dataset**: https://github.com/molecularsets/moses (Molecular SMILES)
