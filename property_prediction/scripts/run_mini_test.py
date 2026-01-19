#!/usr/bin/env python3
"""
Mini test to verify the complete property prediction pipeline.
Uses 100 samples per dataset and tests all featurizers.
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

from datasets import load_lipophilicity, load_bbbp, load_pi1m_tg_mac
from featurizers import RDKitFeaturizer
from featurizers.hnet_featurizer import HNetFeaturizer, MOLECULE_CHECKPOINTS, POLYMER_CHECKPOINTS
from models.xgboost_predictor import XGBoostPredictor


CHECKPOINTS_DIR = Path('/home/ec2-user/hnet_smiles/checkpoints')
RESULTS_DIR = Path(__file__).parent.parent / 'results'
N_SAMPLES = 100
N_FOLDS = 3


def test_molecule_pipeline():
    """Test molecule property prediction pipeline."""
    print("\n" + "="*60)
    print("MOLECULE PIPELINE TEST")
    print("="*60)
    
    results = []
    
    # Load datasets
    lipo = load_lipophilicity(max_samples=N_SAMPLES)
    bbbp = load_bbbp(max_samples=N_SAMPLES)
    
    # Test RDKit featurizer
    print("\n--- RDKit Featurizer ---")
    rdkit = RDKitFeaturizer(feature_type='combined', handle_polymer_smiles=False)
    
    # Lipophilicity (regression)
    print("Lipophilicity (regression):")
    X_lipo = rdkit.featurize_batch(lipo.smiles.tolist(), show_progress=False)
    predictor = XGBoostPredictor(task_type='regression', n_folds=N_FOLDS)
    scores = predictor.train_cv(X_lipo, lipo.targets, verbose=False)
    print(f"  MAE: {scores['mae']:.4f} ± {scores['mae_std']:.4f}")
    results.append({'Model': 'RDKit', 'Dataset': 'Lipophilicity', 'MAE': scores['mae'], 'MAE_std': scores['mae_std']})
    
    # BBBP (classification)
    print("BBBP (classification):")
    X_bbbp = rdkit.featurize_batch(bbbp.smiles.tolist(), show_progress=False)
    predictor = XGBoostPredictor(task_type='classification', n_folds=N_FOLDS)
    scores = predictor.train_cv(X_bbbp, bbbp.targets, verbose=False)
    print(f"  Acc: {scores['accuracy']:.4f} ± {scores['accuracy_std']:.4f}, AUC: {scores['auc']:.4f}")
    results.append({'Model': 'RDKit', 'Dataset': 'BBBP', 'Accuracy': scores['accuracy'], 'AUC': scores['auc']})
    
    # Test H-Net featurizers (only first one for speed)
    ckpt_name = list(MOLECULE_CHECKPOINTS.keys())[0]
    ckpt_path = CHECKPOINTS_DIR / MOLECULE_CHECKPOINTS[ckpt_name]
    
    print(f"\n--- H-Net ({ckpt_name}) ---")
    
    for pooling in ['mean', 'cls']:
        print(f"Pooling: {pooling}")
        hnet = HNetFeaturizer(str(ckpt_path), pooling=pooling)
        
        # Lipophilicity
        X_lipo_hnet = hnet.featurize_batch(lipo.smiles.tolist(), show_progress=True)
        predictor = XGBoostPredictor(task_type='regression', n_folds=N_FOLDS)
        scores = predictor.train_cv(X_lipo_hnet, lipo.targets, verbose=False)
        print(f"  Lipophilicity MAE: {scores['mae']:.4f} ± {scores['mae_std']:.4f}")
        results.append({'Model': f'HNet_{ckpt_name}_{pooling}', 'Dataset': 'Lipophilicity', 'MAE': scores['mae'], 'MAE_std': scores['mae_std']})
        
        # BBBP
        X_bbbp_hnet = hnet.featurize_batch(bbbp.smiles.tolist(), show_progress=True)
        predictor = XGBoostPredictor(task_type='classification', n_folds=N_FOLDS)
        scores = predictor.train_cv(X_bbbp_hnet, bbbp.targets, verbose=False)
        print(f"  BBBP Acc: {scores['accuracy']:.4f}, AUC: {scores['auc']:.4f}")
        results.append({'Model': f'HNet_{ckpt_name}_{pooling}', 'Dataset': 'BBBP', 'Accuracy': scores['accuracy'], 'AUC': scores['auc']})
    
    return results


def test_polymer_pipeline():
    """Test polymer property prediction pipeline."""
    print("\n" + "="*60)
    print("POLYMER PIPELINE TEST")
    print("="*60)
    
    results = []
    
    # Load datasets
    tg_data = load_pi1m_tg_mac(target='Tg', max_samples=N_SAMPLES)
    mac_data = load_pi1m_tg_mac(target='MAC', max_samples=N_SAMPLES)
    
    # Test RDKit featurizer with polymer handling
    print("\n--- RDKit Featurizer (polymer mode) ---")
    rdkit = RDKitFeaturizer(feature_type='combined', handle_polymer_smiles=True)
    
    # Tg (regression)
    print("Tg prediction (regression):")
    X_tg = rdkit.featurize_batch(tg_data.smiles.tolist(), show_progress=False)
    predictor = XGBoostPredictor(task_type='regression', n_folds=N_FOLDS)
    scores = predictor.train_cv(X_tg, tg_data.targets, verbose=False)
    print(f"  MAE: {scores['mae']:.4f} ± {scores['mae_std']:.4f}")
    results.append({'Model': 'RDKit', 'Dataset': 'Tg', 'MAE': scores['mae'], 'MAE_std': scores['mae_std']})
    
    # MAC (regression)
    print("MAC prediction (regression):")
    X_mac = rdkit.featurize_batch(mac_data.smiles.tolist(), show_progress=False)
    predictor = XGBoostPredictor(task_type='regression', n_folds=N_FOLDS)
    scores = predictor.train_cv(X_mac, mac_data.targets, verbose=False)
    print(f"  MAE: {scores['mae']:.4f} ± {scores['mae_std']:.4f}")
    results.append({'Model': 'RDKit', 'Dataset': 'MAC', 'MAE': scores['mae'], 'MAE_std': scores['mae_std']})
    
    # Test H-Net featurizers (only first one for speed)
    ckpt_name = list(POLYMER_CHECKPOINTS.keys())[0]
    ckpt_path = CHECKPOINTS_DIR / POLYMER_CHECKPOINTS[ckpt_name]
    
    print(f"\n--- H-Net ({ckpt_name}) ---")
    
    for pooling in ['mean', 'cls']:
        print(f"Pooling: {pooling}")
        hnet = HNetFeaturizer(str(ckpt_path), pooling=pooling)
        
        # Tg
        X_tg_hnet = hnet.featurize_batch(tg_data.smiles.tolist(), show_progress=True)
        predictor = XGBoostPredictor(task_type='regression', n_folds=N_FOLDS)
        scores = predictor.train_cv(X_tg_hnet, tg_data.targets, verbose=False)
        print(f"  Tg MAE: {scores['mae']:.4f} ± {scores['mae_std']:.4f}")
        results.append({'Model': f'HNet_{ckpt_name}_{pooling}', 'Dataset': 'Tg', 'MAE': scores['mae'], 'MAE_std': scores['mae_std']})
        
        # MAC
        X_mac_hnet = hnet.featurize_batch(mac_data.smiles.tolist(), show_progress=True)
        predictor = XGBoostPredictor(task_type='regression', n_folds=N_FOLDS)
        scores = predictor.train_cv(X_mac_hnet, mac_data.targets, verbose=False)
        print(f"  MAC MAE: {scores['mae']:.4f} ± {scores['mae_std']:.4f}")
        results.append({'Model': f'HNet_{ckpt_name}_{pooling}', 'Dataset': 'MAC', 'MAE': scores['mae'], 'MAE_std': scores['mae_std']})
    
    return results


def main():
    print("="*60)
    print("PROPERTY PREDICTION MINI TEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Samples per dataset: {N_SAMPLES}")
    print(f"CV folds: {N_FOLDS}")
    print("="*60)
    
    all_results = []
    
    # Test molecule pipeline
    mol_results = test_molecule_pipeline()
    all_results.extend(mol_results)
    
    # Test polymer pipeline
    poly_results = test_polymer_pipeline()
    all_results.extend(poly_results)
    
    # Create summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))
    
    # Save results
    results_path = RESULTS_DIR / 'tables' / 'mini_test_results.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n✓ Mini test completed successfully!")


if __name__ == '__main__':
    main()





