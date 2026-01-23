#!/usr/bin/env python3
"""
Upload H-Net SMILES model checkpoints to Hugging Face Hub.

This script uploads all 8 models from the paper:
"Learning Chemical Grammar: Dynamic Tokenization for SMILES with Hierarchical Networks"
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Hugging Face token - set via environment variable or huggingface-cli login
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    try:
        from huggingface_hub import HfFolder
        HF_TOKEN = HfFolder.get_token()
    except:
        pass

if not HF_TOKEN:
    print("Error: No Hugging Face token found.")
    print("Please either:")
    print("  1. Set HF_TOKEN environment variable: export HF_TOKEN=your_token")
    print("  2. Login via CLI: huggingface-cli login")
    sys.exit(1)

# Username for the repository (will be set after authentication)
HF_USERNAME = "jordiferrero"

# Model mapping: paper name -> checkpoint directory
MODELS = {
    "PI1M-68M": {
        "checkpoint_dir": "checkpoints/run_large_20251113_181705",
        "description": "PI1M polymer dataset, 68M bytes (~1 epoch), 10x concatenation, 1-stage architecture",
        "dataset": "PI1M",
        "bytes": "68M",
        "epochs": 1,
        "concat": True,
        "architecture": "1-stage"
    },
    "PI1M-340M": {
        "checkpoint_dir": "checkpoints/run_large_20251111_181836",
        "description": "PI1M polymer dataset, 340M bytes (~5 epochs), 10x concatenation, 1-stage architecture",
        "dataset": "PI1M",
        "bytes": "340M",
        "epochs": 5,
        "concat": True,
        "architecture": "1-stage"
    },
    "PI1M-1B": {
        "checkpoint_dir": "checkpoints/run_large_20251112_150502",
        "description": "PI1M polymer dataset, 1.05B bytes (~22 epochs), 10x concatenation, 1-stage architecture",
        "dataset": "PI1M",
        "bytes": "1.05B",
        "epochs": 22,
        "concat": True,
        "architecture": "1-stage"
    },
    "PI1M-nocat": {
        "checkpoint_dir": "checkpoints/run_large_20251111_075600",
        "description": "PI1M polymer dataset, 340M bytes (~5 epochs), no concatenation, 1-stage architecture",
        "dataset": "PI1M",
        "bytes": "340M",
        "epochs": 5,
        "concat": False,
        "architecture": "1-stage"
    },
    "PI1M-2stg": {
        "checkpoint_dir": "checkpoints/run_large_20260115_191350",
        "description": "PI1M polymer dataset, 340M bytes (~5 epochs), 10x concatenation, 2-stage hierarchical architecture",
        "dataset": "PI1M",
        "bytes": "340M",
        "epochs": 5,
        "concat": True,
        "architecture": "2-stage"
    },
    "MOSES-340M": {
        "checkpoint_dir": "checkpoints/run_large_20251112_071557",
        "description": "MOSES molecular dataset, 340M bytes (~5 epochs), 10x concatenation, 1-stage architecture",
        "dataset": "MOSES",
        "bytes": "340M",
        "epochs": 5,
        "concat": True,
        "architecture": "1-stage"
    },
    "MOSES-nocat": {
        "checkpoint_dir": "checkpoints/run_large_20251113_074900",
        "description": "MOSES molecular dataset, 340M bytes (~5 epochs), no concatenation, 1-stage architecture",
        "dataset": "MOSES",
        "bytes": "340M",
        "epochs": 5,
        "concat": False,
        "architecture": "1-stage"
    },
    "MOSES-2stg": {
        "checkpoint_dir": "checkpoints/run_large_20260116_074355",
        "description": "MOSES molecular dataset, 340M bytes (~5 epochs), 10x concatenation, 2-stage hierarchical architecture",
        "dataset": "MOSES",
        "bytes": "340M",
        "epochs": 5,
        "concat": True,
        "architecture": "2-stage"
    },
}


def create_model_card(model_name: str, model_info: dict, hf_username: str) -> str:
    """Create a model card (README.md) for Hugging Face."""
    concat_str = "10x SMILES per example" if model_info["concat"] else "None (single SMILES)"
    
    return f"""---
license: mit
tags:
- chemistry
- smiles
- tokenization
- dynamic-tokenization
- h-net
- hierarchical-networks
- molecular-representation
- polymer
- mamba
- transformer
datasets:
- {model_info["dataset"]}
language:
- en
pipeline_tag: feature-extraction
---

# {model_name}

**H-Net model for dynamic SMILES tokenization**

{model_info["description"]}

## Model Details

| Property | Value |
|----------|-------|
| **Architecture** | H-Net (Hierarchical Network) |
| **Parameters** | ~350M |
| **Dataset** | {model_info["dataset"]} |
| **Training Bytes** | {model_info["bytes"]} |
| **Training Epochs** | {model_info["epochs"]} |
| **Concatenation** | {concat_str} |
| **Architecture Variant** | {model_info["architecture"]} |

### Architecture Layout

{"1-stage: `['m4', ['T22'], 'm4']`" if model_info["architecture"] == "1-stage" else "2-stage: `['m4', ['T1m4', ['T22'], 'm4T1'], 'm4']`"}

- **Encoder**: 4 Mamba blocks for byte-level encoding
- **Core**: {"22 Transformer blocks with boundary prediction" if model_info["architecture"] == "1-stage" else "2-level hierarchical: Stage 0 (T1+4 Mamba) + Stage 1 (22 Transformer blocks)"}
- **Decoder**: 4 Mamba blocks for final decoding

## Files

- `checkpoints/checkpoint_bytes_best.pt` - Best checkpoint (lowest validation loss)
- `checkpoints/checkpoint_epoch_*.pt` - Epoch checkpoints
- `metadata.json` - Training configuration and history
- `test_smiles.txt` - Test SMILES used during training
- `visualizations/` - Training evolution GIFs and prediction files

## Usage

```python
import torch
from pathlib import Path

# Load checkpoint
checkpoint_path = "checkpoints/checkpoint_bytes_best.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# The checkpoint contains:
# - 'model_state_dict': Model weights
# - 'optimizer_state_dict': Optimizer state
# - 'epoch': Training epoch
# - 'metrics': Training metrics
# - 'cumulative_training_bytes': Total bytes processed

# Load into your H-Net model
# model.load_state_dict(checkpoint['model_state_dict'])
```

## Performance

### Tokenization Metrics (from paper)

| Metric | Value |
|--------|-------|
| Bits-per-byte (BPB) | {"0.64" if model_name == "PI1M-1B" else "0.69" if "340M" in model_name else "0.83"} |
| Mean token length | {"2.9" if model_name == "PI1M-1B" else "2.6" if "PI1M" in model_name else "2.0"} |

### Property Prediction (embeddings)

H-Net embeddings outperform RDKit descriptors on classification tasks:
- BBBP: 0.950 AUC (vs 0.927 for RDKit)
- HIV: 0.788 AUC (vs 0.760 for RDKit)

## Citation

```bibtex
@inproceedings{{hnet_smiles_2026,
  title={{Learning Chemical Grammar: Dynamic Tokenization for SMILES with Hierarchical Networks}},
  author={{Anonymous}},
  booktitle={{International Conference on Machine Learning (ICML)}},
  year={{2026}}
}}
```

## Related Models

All models from the paper are available:

**Polymer (PI1M) Models:**
- [PI1M-68M](https://huggingface.co/{hf_username}/PI1M-68M) - 1 epoch, with concatenation
- [PI1M-340M](https://huggingface.co/{hf_username}/PI1M-340M) - 5 epochs, with concatenation  
- [PI1M-1B](https://huggingface.co/{hf_username}/PI1M-1B) - 22 epochs, with concatenation (best compression)
- [PI1M-nocat](https://huggingface.co/{hf_username}/PI1M-nocat) - 5 epochs, no concatenation
- [PI1M-2stg](https://huggingface.co/{hf_username}/PI1M-2stg) - 5 epochs, 2-stage architecture

**Molecular (MOSES) Models:**
- [MOSES-340M](https://huggingface.co/{hf_username}/MOSES-340M) - 5 epochs, with concatenation
- [MOSES-nocat](https://huggingface.co/{hf_username}/MOSES-nocat) - 5 epochs, no concatenation
- [MOSES-2stg](https://huggingface.co/{hf_username}/MOSES-2stg) - 5 epochs, 2-stage architecture

## License

MIT License
"""


def upload_model(model_name: str, model_info: dict, api: HfApi, base_path: Path, hf_username: str):
    """Upload a single model to Hugging Face using large folder upload."""
    repo_id = f"{hf_username}/{model_name}"
    checkpoint_dir = base_path / model_info["checkpoint_dir"]
    
    if not checkpoint_dir.exists():
        print(f"❌ Error: {checkpoint_dir} not found, skipping {model_name}")
        return False
    
    print(f"\n{'='*60}")
    print(f"📤 Uploading {model_name}...")
    print(f"   Source: {checkpoint_dir}")
    print(f"   Destination: https://huggingface.co/{repo_id}")
    print(f"{'='*60}")
    
    # Create repository
    try:
        create_repo(repo_id, exist_ok=True, private=False, token=HF_TOKEN, repo_type="model")
        print(f"   ✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"   ⚠️  Repository creation note: {e}")
    
    # Create model card
    model_card = create_model_card(model_name, model_info, hf_username)
    readme_path = checkpoint_dir / "README.md"
    readme_path.write_text(model_card)
    print(f"   ✓ Model card created")
    
    # Upload using large folder method (handles large files properly)
    try:
        print(f"   📦 Starting upload (this may take a while for large checkpoints)...")
        # Use api instance method which already has the token
        api.upload_large_folder(
            folder_path=str(checkpoint_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["*.pyc", "__pycache__", ".git*"],
            print_report=True,
            print_report_every=30,
        )
        print(f"   ✅ Successfully uploaded to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to upload all models."""
    print("=" * 60)
    print("H-Net SMILES Model Upload to Hugging Face")
    print("=" * 60)
    
    # Initialize API
    api = HfApi(token=HF_TOKEN)
    
    # Get base path
    base_path = Path(__file__).parent
    
    # Verify token and get username
    try:
        user_info = api.whoami()
        hf_username = user_info['name']
        print(f"\n✓ Authenticated as: {hf_username}")
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        print("Please check your HF_TOKEN")
        return
    
    # Check which model to upload (can specify on command line)
    if len(sys.argv) > 1:
        model_filter = sys.argv[1]
        models_to_upload = {k: v for k, v in MODELS.items() if model_filter.lower() in k.lower()}
        if not models_to_upload:
            print(f"No models matching '{model_filter}' found.")
            print(f"Available models: {', '.join(MODELS.keys())}")
            return
    else:
        models_to_upload = MODELS
    
    print(f"\nModels to upload: {', '.join(models_to_upload.keys())}")
    
    # Upload each model
    successful = 0
    failed = 0
    
    for model_name, model_info in models_to_upload.items():
        if upload_model(model_name, model_info, api, base_path, hf_username):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"\nModels available at: https://huggingface.co/{hf_username}")
    print("=" * 60)


if __name__ == "__main__":
    main()
