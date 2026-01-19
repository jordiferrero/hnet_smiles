#!/usr/bin/env python3
"""
Extract features from all featurizers and cache to disk.

This script extracts features using:
1. H-Net models (all checkpoints, all pooling strategies)
2. RDKit descriptors

Features are cached to results/features/ for fast experiment iteration.

Usage:
    python scripts/extract_all_features.py --dataset polymer
    python scripts/extract_all_features.py --dataset molecule
    python scripts/extract_all_features.py --dataset all
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_pi1m_tg_mac, load_lipophilicity, load_bbbp
from featurizers import RDKitFeaturizer

# H-Net imports (needs GPU)
try:
    from featurizers.hnet_featurizer import (
        HNetFeaturizer,
        POLYMER_CHECKPOINTS,
        MOLECULE_CHECKPOINTS,
    )
    HNET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: H-Net featurizer not available: {e}")
    HNET_AVAILABLE = False


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'
FEATURES_DIR = Path(__file__).parent.parent / 'results' / 'features'


def extract_rdkit_features(
    smiles: np.ndarray,
    dataset_name: str,
    for_polymers: bool = False,
) -> dict:
    """Extract RDKit features and save to cache."""
    output_dir = FEATURES_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = output_dir / 'rdkit_combined.npz'
    
    if cache_path.exists():
        print(f"  Loading cached RDKit features from {cache_path}")
        data = np.load(cache_path)
        return {'features': data['features'], 'feature_names': data['feature_names'].tolist()}
    
    print(f"  Extracting RDKit features (combined)...")
    featurizer = RDKitFeaturizer(
        feature_type='combined',
        handle_polymer_smiles=for_polymers,
    )
    
    features = featurizer.featurize_batch(smiles.tolist(), show_progress=True)
    feature_names = featurizer.get_feature_names()
    
    # Save to cache
    np.savez_compressed(
        cache_path,
        features=features,
        feature_names=np.array(feature_names),
    )
    print(f"  Saved to {cache_path}")
    
    return {'features': features, 'feature_names': feature_names}


def extract_hnet_features(
    smiles: np.ndarray,
    dataset_name: str,
    checkpoint_name: str,
    checkpoint_dir: str,
    pooling: str = 'mean',
    device: str = 'cuda',
) -> dict:
    """Extract H-Net features and save to cache."""
    output_dir = FEATURES_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = output_dir / f'hnet_{checkpoint_name}_{pooling}.npz'
    
    if cache_path.exists():
        print(f"  Loading cached H-Net features from {cache_path}")
        data = np.load(cache_path)
        return {'features': data['features'], 'feature_names': data['feature_names'].tolist()}
    
    print(f"  Extracting H-Net features ({checkpoint_name}, {pooling})...")
    
    featurizer = HNetFeaturizer(
        checkpoint_dir=checkpoint_dir,
        pooling=pooling,
        device=device,
    )
    
    features = featurizer.featurize_batch(smiles.tolist(), show_progress=True)
    feature_names = featurizer.get_feature_names()
    
    # Save to cache
    np.savez_compressed(
        cache_path,
        features=features,
        feature_names=np.array(feature_names),
    )
    print(f"  Saved to {cache_path}")
    
    return {'features': features, 'feature_names': feature_names}


def extract_polymer_features(device: str = 'cuda', max_samples: int = None):
    """Extract all features for polymer dataset."""
    print("\n" + "=" * 60)
    print("Extracting features for POLYMER dataset (PI1M_Tg_MAC)")
    print("=" * 60)
    
    # Load dataset (we only need SMILES for feature extraction)
    dataset = load_pi1m_tg_mac(target='Tg', max_samples=max_samples)
    smiles = dataset.smiles
    
    # Save SMILES order for later matching
    output_dir = FEATURES_DIR / 'polymer'
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'smiles.npy', smiles)
    np.save(output_dir / 'targets_Tg.npy', dataset.targets)
    
    # Load MAC targets too
    dataset_mac = load_pi1m_tg_mac(target='MAC', max_samples=max_samples)
    np.save(output_dir / 'targets_MAC.npy', dataset_mac.targets)
    
    # 1. RDKit features
    print("\n1. RDKit features")
    extract_rdkit_features(smiles, 'polymer', for_polymers=True)
    
    # 2. H-Net features (all checkpoints, all pooling strategies)
    if HNET_AVAILABLE:
        print("\n2. H-Net features")
        
        for pooling in ['mean', 'max', 'cls']:
            print(f"\n  Pooling: {pooling}")
            
            for name, run_dir in POLYMER_CHECKPOINTS.items():
                checkpoint_path = CHECKPOINTS_DIR / run_dir
                
                if not checkpoint_path.exists():
                    print(f"    Skipping {name}: checkpoint not found")
                    continue
                
                try:
                    extract_hnet_features(
                        smiles=smiles,
                        dataset_name='polymer',
                        checkpoint_name=name,
                        checkpoint_dir=str(checkpoint_path),
                        pooling=pooling,
                        device=device,
                    )
                except Exception as e:
                    print(f"    Error extracting {name}: {e}")
    
    print("\nPolymer feature extraction complete!")


def extract_molecule_features(device: str = 'cuda', max_samples: int = None):
    """Extract all features for molecule datasets."""
    print("\n" + "=" * 60)
    print("Extracting features for MOLECULE datasets")
    print("=" * 60)
    
    # Load Lipophilicity
    print("\n--- Lipophilicity ---")
    try:
        dataset_lipo = load_lipophilicity(max_samples=max_samples)
        smiles_lipo = dataset_lipo.smiles
        
        output_dir = FEATURES_DIR / 'lipophilicity'
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / 'smiles.npy', smiles_lipo)
        np.save(output_dir / 'targets.npy', dataset_lipo.targets)
        
        # RDKit
        print("\n1. RDKit features (Lipophilicity)")
        extract_rdkit_features(smiles_lipo, 'lipophilicity', for_polymers=False)
        
        # H-Net
        if HNET_AVAILABLE:
            print("\n2. H-Net features (Lipophilicity)")
            for pooling in ['mean', 'max', 'cls']:
                for name, run_dir in MOLECULE_CHECKPOINTS.items():
                    checkpoint_path = CHECKPOINTS_DIR / run_dir
                    if checkpoint_path.exists():
                        try:
                            extract_hnet_features(
                                smiles=smiles_lipo,
                                dataset_name='lipophilicity',
                                checkpoint_name=name,
                                checkpoint_dir=str(checkpoint_path),
                                pooling=pooling,
                                device=device,
                            )
                        except Exception as e:
                            print(f"    Error: {e}")
    except FileNotFoundError as e:
        print(f"  Skipping Lipophilicity: {e}")
    
    # Load BBBP
    print("\n--- BBBP ---")
    try:
        dataset_bbbp = load_bbbp(max_samples=max_samples)
        smiles_bbbp = dataset_bbbp.smiles
        
        output_dir = FEATURES_DIR / 'bbbp'
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / 'smiles.npy', smiles_bbbp)
        np.save(output_dir / 'targets.npy', dataset_bbbp.targets)
        
        # RDKit
        print("\n1. RDKit features (BBBP)")
        extract_rdkit_features(smiles_bbbp, 'bbbp', for_polymers=False)
        
        # H-Net
        if HNET_AVAILABLE:
            print("\n2. H-Net features (BBBP)")
            for pooling in ['mean', 'max', 'cls']:
                for name, run_dir in MOLECULE_CHECKPOINTS.items():
                    checkpoint_path = CHECKPOINTS_DIR / run_dir
                    if checkpoint_path.exists():
                        try:
                            extract_hnet_features(
                                smiles=smiles_bbbp,
                                dataset_name='bbbp',
                                checkpoint_name=name,
                                checkpoint_dir=str(checkpoint_path),
                                pooling=pooling,
                                device=device,
                            )
                        except Exception as e:
                            print(f"    Error: {e}")
    except FileNotFoundError as e:
        print(f"  Skipping BBBP: {e}")
    
    print("\nMolecule feature extraction complete!")


def main():
    parser = argparse.ArgumentParser(description='Extract features for property prediction')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['polymer', 'molecule', 'all'],
        default='all',
        help='Which dataset(s) to process',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for H-Net inference (cuda or cpu)',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to process (for testing)',
    )
    
    args = parser.parse_args()
    
    print(f"Feature Extraction - {datetime.now().isoformat()}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples or 'all'}")
    
    if args.dataset in ['polymer', 'all']:
        extract_polymer_features(device=args.device, max_samples=args.max_samples)
    
    if args.dataset in ['molecule', 'all']:
        extract_molecule_features(device=args.device, max_samples=args.max_samples)
    
    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print(f"Features saved to: {FEATURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()





