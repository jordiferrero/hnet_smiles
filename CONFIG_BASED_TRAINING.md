# Config-Based Training

All training parameters are now configured through the config JSON files. No command-line arguments needed!

## Usage

### Simple Usage (Config File Only)

```bash
# All parameters come from config file
python train_smiles.py --config configs/hnet_smiles_small.json
```

### Optional Override

```bash
# Config file provides defaults, but you can override specific parameters
python train_smiles.py --config configs/hnet_smiles_small.json --resume checkpoints/run_small_20251107_123456/checkpoints/checkpoint_epoch_5.pt
```

## Config File Structure

Each config file now contains a `training_config` section with all training parameters:

```json
{
  "arch_layout": ["m4", ["T22"], "m4"],
  "d_model": [512, 768],
  "d_intermediate": [0, 2048],
  "vocab_size": 256,
  "ssm_cfg": { ... },
  "attn_cfg": { ... },
  "tie_embeddings": false,
  "training_config": {
    "data": "datasets/PI1M/PI1M_v2.csv",
    "phase": "small",
    "max_samples": 1000,
    "batch_size": 8,
    "epochs": 5,
    "lr": 0.0001,
    "weight_decay": 0.1,
    "gradient_accumulation": 1,
    "concatenate": true,
    "num_concatenate": 3,
    "checkpoint_bytes": 1000000,
    "num_test_samples": 50,
    "num_visualize": 5,
    "skip_visualization": false,
    "output_dir": "checkpoints",
    "no_amp": false
  }
}
```

## Training Config Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `data` | string | Path to SMILES CSV file | `datasets/PI1M/PI1M_v2.csv` |
| `phase` | string | Training phase (`small`, `medium`, `large`) | `small` |
| `max_samples` | int/null | Maximum samples to use (null = all) | `null` |
| `batch_size` | int | Batch size | `8` |
| `epochs` | int | Number of epochs | `10` |
| `lr` | float | Learning rate | `0.0001` |
| `weight_decay` | float | Weight decay | `0.1` |
| `gradient_accumulation` | int | Gradient accumulation steps | `1` |
| `concatenate` | bool | Concatenate multiple SMILES | `true` |
| `num_concatenate` | int | Number of SMILES to concatenate | `3` |
| `checkpoint_bytes` | int | Save checkpoint every N bytes | `1000000` |
| `num_test_samples` | int | Number of SMILES for testing | `50` |
| `num_visualize` | int | Number of test SMILES to visualize | `5` |
| `skip_visualization` | bool | Skip automatic visualization | `false` |
| `output_dir` | string | Output directory for checkpoints | `checkpoints` |
| `no_amp` | bool | Disable automatic mixed precision | `false` |

## Config Files

### Small Config (`hnet_smiles_small.json`)
- Phase: `small`
- Max samples: `1000`
- Batch size: `8`
- Epochs: `5`

### Medium Config (`hnet_smiles_medium.json`)
- Phase: `medium`
- Max samples: `10000`
- Batch size: `16`
- Epochs: `10`

### Large Config (`hnet_smiles_large.json`)
- Phase: `large`
- Max samples: `null` (all)
- Batch size: `32`
- Epochs: `20`

## Examples

### Run Small Training
```bash
python train_smiles.py --config configs/hnet_smiles_small.json
```

### Run Medium Training
```bash
python train_smiles.py --config configs/hnet_smiles_medium.json
```

### Run Large Training
```bash
python train_smiles.py --config configs/hnet_smiles_large.json
```

### Resume Training
```bash
python train_smiles.py --config configs/hnet_smiles_small.json --resume checkpoints/run_small_20251107_123456/checkpoints/checkpoint_epoch_3.pt
```

## Benefits

1. **Single Source of Truth**: All parameters in one config file
2. **Reproducible**: Config file can be version controlled
3. **Easy to Share**: Share config files instead of long command lines
4. **No Command-Line Arguments Needed**: Just specify the config file
5. **Still Flexible**: Command-line arguments can override config values if needed

## Notes

- The `--config` argument is **required**
- The `--resume` argument is **optional** (for resuming training)
- All other parameters come from the config file
- Command-line arguments can still override config values if provided
- Config files are self-contained and include all training parameters

