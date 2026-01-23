# Model Checkpoints

⚠️ **Large checkpoint files are hosted on Hugging Face Hub**

Due to the large size of model checkpoints (~8GB per model, ~200GB+ total), they are not stored in this GitHub repository.

## Download Checkpoints

All trained model checkpoints from the paper are available on Hugging Face:

### 🔗 [H-Net SMILES Checkpoints Collection](https://huggingface.co/collections/jordiferrero/h-net-smiles-checkpoints-6973301c4e3d748569c70b98)

## Available Models

### Polymer Models (PI1M Dataset)

| Model | Description | Hugging Face Link |
|-------|-------------|-------------------|
| **PI1M-68M** | 68M bytes (~1 epoch), 10x concatenation | [Download](https://huggingface.co/jordiferrero/PI1M-68M) |
| **PI1M-340M** | 340M bytes (~5 epochs), 10x concatenation | [Download](https://huggingface.co/jordiferrero/PI1M-340M) |
| **PI1M-1B** | 1.05B bytes (~22 epochs), 10x concatenation | [Download](https://huggingface.co/jordiferrero/PI1M-1B) |
| **PI1M-nocat** | 340M bytes (~5 epochs), no concatenation | [Download](https://huggingface.co/jordiferrero/PI1M-nocat) |
| **PI1M-2stg** | 340M bytes, 10x concat, 2-stage architecture | [Download](https://huggingface.co/jordiferrero/PI1M-2stg) |

### Molecular Models (MOSES Dataset)

| Model | Description | Hugging Face Link |
|-------|-------------|-------------------|
| **MOSES-340M** | 340M bytes (~5 epochs), 10x concatenation | [Download](https://huggingface.co/jordiferrero/MOSES-340M) |
| **MOSES-nocat** | 340M bytes (~5 epochs), no concatenation | [Download](https://huggingface.co/jordiferrero/MOSES-nocat) |
| **MOSES-2stg** | 340M bytes, 10x concat, 2-stage architecture | [Download](https://huggingface.co/jordiferrero/MOSES-2stg) |

## Quick Download with Python

```python
from huggingface_hub import hf_hub_download

# Download best checkpoint for PI1M-1B model
checkpoint_path = hf_hub_download(
    repo_id="jordiferrero/PI1M-1B",
    filename="checkpoints/checkpoint_bytes_best.pt"
)

# Load checkpoint
import torch
checkpoint = torch.load(checkpoint_path, map_location="cpu")
```

## Quick Download with CLI

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download entire model folder
huggingface-cli download jordiferrero/PI1M-1B --local-dir ./checkpoints/PI1M-1B

# Or download specific checkpoint
huggingface-cli download jordiferrero/PI1M-1B checkpoints/checkpoint_bytes_best.pt
```

## Checkpoint Contents

Each model repository contains:

```
├── checkpoints/
│   ├── checkpoint_bytes_best.pt    # Best model (lowest loss)
│   └── checkpoint_epoch_*.pt       # Epoch checkpoints
├── metadata.json                    # Training configuration & history
├── test_smiles.txt                  # Test SMILES used during training
├── visualizations/
│   ├── training_evolution_*.gif    # Tokenization evolution animations
│   └── predictions/                # Prediction files at various steps
└── README.md                        # Model card
```

## Citation

If you use these models, please cite:

```bibtex
@inproceedings{hnet_smiles_2026,
  title={Learning Chemical Grammar: Dynamic Tokenization for SMILES with Hierarchical Networks},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```


