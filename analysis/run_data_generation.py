#!/usr/bin/env python3
"""
Complete data generation script for H-Net SMILES tokenization analysis.
Runs SmilesPE benchmark and H-Net inference on all 6 models.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.utils.inference import (
    load_model, run_tokenization_inference, 
    save_tokenization_results, get_model_info
)
from analysis.utils.statistics import compute_token_statistics
from analysis.utils.benchmark import SmilesPEBenchmark

# ============================================================================
# SUBSET MODE - Use a subset for faster but still statistically robust results
# ============================================================================
USE_SUBSET = True
MAX_SAMPLES_SUBSET = 10000  # 10K samples per dataset (~7 hours total, statistically robust)

print("="*80)
print("H-Net SMILES Tokenization Analysis - Data Generation")
if USE_SUBSET:
    print(f"*** SUBSET MODE: Processing {MAX_SAMPLES_SUBSET} samples per dataset for robust statistics ***")
print("="*80)

# Define all models
models = [
    {
        'name': 'PI1M_concat_1epoch',
        'path': project_root / 'checkpoints' / 'run_large_20251113_181705',
        'description': 'PI1M with 10-PSMILES concatenation, 1 epoch (68M bytes)',
    },
    {
        'name': 'PI1M_noconcat_5epoch',
        'path': project_root / 'checkpoints' / 'run_large_20251111_075600',
        'description': 'PI1M no concatenation, 5 epoch (240M bytes)',
    },
    {
        'name': 'PI1M_concat_5epoch',
        'path': project_root / 'checkpoints' / 'run_large_20251111_181836',
        'description': 'PI1M with 10-PSMILES concatenation, 5 epoch (240M bytes)',
    },
    {
        'name': 'PI1M_concat_22epoch',
        'path': project_root / 'checkpoints' / 'run_large_20251112_150502',
        'description': 'PI1M with 10-PSMILES concatenation, 22 epoch (1B bytes)',
    },
    {
        'name': 'MOSES_noconcat_5epoch',
        'path': project_root / 'checkpoints' / 'run_large_20251113_074900',
        'description': 'MOSES no concatenation, 5 epoch (360M bytes)',
    },
    {
        'name': 'MOSES_concat_5epoch',
        'path': project_root / 'checkpoints' / 'run_large_20251112_071557',
        'description': 'MOSES with 10-SMILES concatenation, 5 epoch (360M bytes)',
    },
    # 2-Stage Architecture
    {
        'name': 'PI1M_concat_5epoch_2stage',
        'path': project_root / 'checkpoints' / 'run_large_20260115_191350',
        'description': 'PI1M 2-stage with 10-PSMILES concatenation, 5 epoch (240M bytes)',
    },
    {
        'name': 'MOSES_concat_5epoch_2stage',
        'path': project_root / 'checkpoints' / 'run_large_20260116_074355',
        'description': 'MOSES 2-stage with 10-SMILES concatenation, 5 epoch (360M bytes)',
    },
]

print(f"\nTotal models to process: {len(models)}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# ============================================================================
# STEP 1: Run SmilesPE Benchmark (no GPU needed)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Running SmilesPE Benchmark")
print("="*80)

vocab_path = project_root / 'analysis' / 'data' / 'SPE_ChEMBL.txt'
spe = SmilesPEBenchmark(str(vocab_path))

datasets = [
    {
        'name': 'PI1M',
        'csv': project_root / 'datasets' / 'PI1M' / 'PI1M_v2.csv',
        'type': 'PI1M',
    },
    {
        'name': 'MOSES',
        'csv': project_root / 'datasets' / 'moses' / 'smiles-molecules-moses_all.csv',
        'type': 'MOSES',
    },
]

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Processing SmilesPE for: {dataset['name']}")
    print(f"{'='*60}")
    
    try:
        # Run tokenization
        results = spe.tokenize_dataset(
            str(dataset['csv']),
            dataset['type'],
            max_samples=MAX_SAMPLES_SUBSET if USE_SUBSET else None,
        )
        
        # Save results
        output_path = project_root / 'analysis' / 'data' / 'smilesPE_results' / f"SmilesPE_{dataset['name']}_tokenization.pkl"
        spe.save_results(results, str(output_path))
        
        # Compute statistics
        stats = compute_token_statistics(results)
        
        # Save statistics
        stats_path = project_root / 'analysis' / 'data' / 'statistics' / f"SmilesPE_{dataset['name']}_stats.json"
        stats.save(str(stats_path))
        
        print(f"\n✓ Results saved to: {output_path}")
        print(f"✓ Statistics saved to: {stats_path}")
        print(f"\nSummary:")
        for key, value in stats.get_summary().items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"✗ Error processing {dataset['name']}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# STEP 2: Run H-Net Inference (requires GPU)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Running H-Net Inference on All Models")
print("="*80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

for i, model_config in enumerate(models, 1):
    print(f"\n{'='*60}")
    print(f"Processing Model {i}/{len(models)}: {model_config['name']}")
    print(f"Description: {model_config['description']}")
    print(f"{'='*60}")
    
    try:
        # Load model
        print("Loading model...")
        model, info = load_model(str(model_config['path']), device=device)
        
        # Run tokenization inference
        print(f"Running inference on {info['dataset_type']} dataset...")
        results = run_tokenization_inference(
            model=model,
            dataset_csv=info['dataset_csv'],
            dataset_type=info['dataset_type'],
            device=device,
            max_samples=MAX_SAMPLES_SUBSET if USE_SUBSET else None,
        )
        
        # Save results
        output_path = project_root / 'analysis' / 'data' / 'hnet_results' / f"{model_config['name']}_tokenization.pkl"
        save_tokenization_results(results, str(output_path))
        
        # Compute statistics
        print("Computing statistics...")
        stats = compute_token_statistics(results)
        
        # Save statistics
        stats_path = project_root / 'analysis' / 'data' / 'statistics' / f"{model_config['name']}_stats.json"
        stats.save(str(stats_path))
        
        print(f"\n✓ Results saved to: {output_path}")
        print(f"✓ Statistics saved to: {stats_path}")
        print(f"\nSummary:")
        for key, value in stats.get_summary().items():
            print(f"  {key}: {value}")
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")
        
    except Exception as e:
        print(f"✗ Error processing {model_config['name']}: {e}")
        import traceback
        traceback.print_exc()
        # Try to free GPU memory even on error
        try:
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
        except:
            pass

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DATA GENERATION COMPLETE!")
print("="*80)

# Check what was generated
import os

hnet_results_dir = project_root / 'analysis' / 'data' / 'hnet_results'
spe_results_dir = project_root / 'analysis' / 'data' / 'smilesPE_results'
stats_dir = project_root / 'analysis' / 'data' / 'statistics'

print("\nGenerated Files:")
print("-" * 80)

print("\nH-Net Results:")
if hnet_results_dir.exists():
    files = sorted(hnet_results_dir.glob("*.pkl"))
    for f in files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  ✓ {f.name} ({size_mb:.2f} MB)")
else:
    print("  ⚠️ Directory not found")

print("\nSmilesPE Results:")
if spe_results_dir.exists():
    files = sorted(spe_results_dir.glob("*.pkl"))
    for f in files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  ✓ {f.name} ({size_mb:.2f} MB)")
else:
    print("  ⚠️ Directory not found")

print("\nStatistics:")
if stats_dir.exists():
    files = sorted(stats_dir.glob("*.json"))
    for f in files:
        print(f"  ✓ {f.name}")
else:
    print("  ⚠️ Directory not found")

print("\n" + "="*80)
print("Next steps:")
print("1. Run analysis notebooks 02-05 to generate comparative analyses")
print("2. Check analysis/figures/ for generated visualizations")
print("3. Review analysis/data/ for summary CSV files")
print("="*80)

