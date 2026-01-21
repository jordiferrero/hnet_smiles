#!/usr/bin/env python3
"""
Run Extended MoleculeNet Benchmark Experiments.

This script runs property prediction experiments on 6 additional MoleculeNet
datasets using the molecular H-Net model (MOSES-trained).

New Datasets:
- ESOL (regression): Aqueous solubility
- FreeSolv (regression): Hydration free energy
- HIV (classification): HIV replication inhibition
- BACE (classification): Beta-secretase 1 inhibitors
- ClinTox (classification): Clinical trial toxicity
- Tox21 (multi-label classification): Toxicity panels

Uses existing infrastructure:
- HNetFeaturizer from featurizers/hnet_featurizer.py
- RDKitFeaturizer from featurizers/rdkit_featurizer.py
- XGBoostPredictor from models/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import urllib.request
import json

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, f1_score
)

# Paths
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = SCRIPTS_DIR.parent / 'data' / 'molecule'
RESULTS_DIR = SCRIPTS_DIR.parent / 'results' / 'tables'
CHECKPOINTS_DIR = SCRIPTS_DIR.parent.parent / 'checkpoints'

# Dataset configurations
EXTENDED_DATASETS = {
    'esol': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
        'filename': 'esol.csv',
        'smiles_col': 'smiles',
        'target_col': 'measured log solubility in mols per litre',
        'task': 'regression',
        'metric': 'rmse',
    },
    'freesolv': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
        'filename': 'freesolv.csv',
        'smiles_col': 'smiles',
        'target_col': 'expt',
        'task': 'regression',
        'metric': 'rmse',
    },
    'hiv': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv',
        'filename': 'hiv.csv',
        'smiles_col': 'smiles',
        'target_col': 'HIV_active',
        'task': 'classification',
        'metric': 'auc',
    },
    'bace': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
        'filename': 'bace.csv',
        'smiles_col': 'mol',
        'target_col': 'Class',
        'task': 'classification',
        'metric': 'auc',
    },
    'clintox': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz',
        'filename': 'clintox.csv',
        'smiles_col': 'smiles',
        'target_col': 'CT_TOX',  # Use clinical trial toxicity
        'task': 'classification',
        'metric': 'auc',
    },
    'tox21': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
        'filename': 'tox21.csv',
        'smiles_col': 'smiles',
        'target_col': 'NR-AR',  # Use one representative task
        'task': 'classification',
        'metric': 'auc',
    },
}

# Molecular H-Net checkpoint (MOSES-trained)
HNET_CHECKPOINT = 'run_large_20251112_071557'


def download_dataset(name: str) -> pd.DataFrame:
    """Download and cache a MoleculeNet dataset."""
    info = EXTENDED_DATASETS[name]
    dest_path = DATA_DIR / info['filename']
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not dest_path.exists():
        print(f"Downloading {name}...")
        url = info['url']
        
        if url.endswith('.gz'):
            import gzip
            import shutil
            
            gz_path = dest_path.with_suffix('.csv.gz')
            urllib.request.urlretrieve(url, gz_path)
            
            with gzip.open(gz_path, 'rb') as f_in:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
        else:
            urllib.request.urlretrieve(url, dest_path)
        
        print(f"  Saved to {dest_path}")
    
    df = pd.read_csv(dest_path)
    return df


def load_dataset(name: str) -> tuple:
    """Load dataset and return SMILES and targets."""
    info = EXTENDED_DATASETS[name]
    df = download_dataset(name)
    
    smiles_col = info['smiles_col']
    target_col = info['target_col']
    
    # Filter valid entries
    df = df[[smiles_col, target_col]].dropna()
    
    smiles = df[smiles_col].tolist()
    targets = df[target_col].values
    
    print(f"Loaded {name}: {len(smiles)} samples")
    
    return smiles, targets, info['task'], info['metric']


def extract_hnet_features(smiles_list: list, checkpoint_dir: Path, pooling: str = 'mean') -> np.ndarray:
    """Extract H-Net features for a list of SMILES."""
    # Import from the property_prediction featurizers
    import sys
    featurizer_path = str(Path(__file__).parent.parent / 'featurizers')
    if featurizer_path not in sys.path:
        sys.path.insert(0, featurizer_path)
    
    # Ensure hnet module is available
    hnet_repo = str(Path(__file__).parent.parent.parent / 'original_resources' / 'hnet-github-repo')
    if hnet_repo not in sys.path:
        sys.path.insert(0, hnet_repo)
    
    from hnet_featurizer import HNetFeaturizer
    
    featurizer = HNetFeaturizer(
        checkpoint_dir=str(checkpoint_dir),
        pooling=pooling,
        device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
    )
    
    features = featurizer.featurize_batch(smiles_list, show_progress=True)
    return features


def extract_rdkit_features(smiles_list: list) -> np.ndarray:
    """Extract RDKit features for a list of SMILES."""
    # Import here to avoid dependency issues
    import sys
    featurizer_path = str(Path(__file__).parent.parent / 'featurizers')
    if featurizer_path not in sys.path:
        sys.path.insert(0, featurizer_path)
    
    from rdkit_featurizer import RDKitFeaturizer
    
    featurizer = RDKitFeaturizer()
    features = featurizer.featurize_batch(smiles_list, show_progress=True)
    return features


def run_cv_experiment(X: np.ndarray, y: np.ndarray, task: str, n_folds: int = 5) -> dict:
    """Run cross-validation experiment."""
    from xgboost import XGBClassifier, XGBRegressor
    
    # Handle NaN in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if task == 'regression':
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        mae_scores = []
        rmse_scores = []
        r2_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2_scores.append(r2_score(y_test, y_pred))
        
        return {
            'mae': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'rmse': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'r2': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'fold_scores': mae_scores,
        }
    
    else:  # classification
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
        )
        
        # Handle class imbalance
        y_int = y.astype(int)
        
        try:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(kf.split(X, y_int))
        except ValueError:
            # Fall back to regular KFold if stratification fails
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(kf.split(X))
        
        acc_scores = []
        auc_scores = []
        
        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_int[train_idx], y_int[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            acc_scores.append(accuracy_score(y_test, y_pred))
            
            try:
                auc_scores.append(roc_auc_score(y_test, y_prob))
            except ValueError:
                auc_scores.append(0.5)  # Default for single-class fold
        
        return {
            'accuracy': np.mean(acc_scores),
            'accuracy_std': np.std(acc_scores),
            'auc': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'fold_scores': auc_scores,
        }


def run_dataset_experiments(dataset_name: str) -> list:
    """Run all experiments for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load dataset
    smiles, targets, task, metric = load_dataset(dataset_name)
    
    results = []
    
    # Limit to first 5000 samples for large datasets
    max_samples = 5000
    if len(smiles) > max_samples:
        print(f"Limiting to {max_samples} samples (from {len(smiles)})")
        indices = np.random.RandomState(42).choice(len(smiles), max_samples, replace=False)
        smiles = [smiles[i] for i in indices]
        targets = targets[indices]
    
    # 1. RDKit baseline
    print("\n--- RDKit Features ---")
    try:
        X_rdkit = extract_rdkit_features(smiles)
        rdkit_results = run_cv_experiment(X_rdkit, targets, task)
        rdkit_results['model'] = 'RDKit'
        rdkit_results['dataset'] = dataset_name
        rdkit_results['task'] = task
        results.append(rdkit_results)
        
        if task == 'regression':
            print(f"  RMSE: {rdkit_results['rmse']:.4f} ± {rdkit_results['rmse_std']:.4f}")
        else:
            print(f"  AUC: {rdkit_results['auc']:.4f} ± {rdkit_results['auc_std']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 2. H-Net (molecular model)
    print("\n--- H-Net Features (mean pooling) ---")
    checkpoint_path = CHECKPOINTS_DIR / HNET_CHECKPOINT
    
    if checkpoint_path.exists():
        try:
            X_hnet = extract_hnet_features(smiles, checkpoint_path, pooling='mean')
            hnet_results = run_cv_experiment(X_hnet, targets, task)
            hnet_results['model'] = 'H-Net (mean)'
            hnet_results['dataset'] = dataset_name
            hnet_results['task'] = task
            results.append(hnet_results)
            
            if task == 'regression':
                print(f"  RMSE: {hnet_results['rmse']:.4f} ± {hnet_results['rmse_std']:.4f}")
            else:
                print(f"  AUC: {hnet_results['auc']:.4f} ± {hnet_results['auc_std']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  Checkpoint not found: {checkpoint_path}")
    
    return results


def main():
    """Run all extended MoleculeNet experiments."""
    parser = argparse.ArgumentParser(description='Run extended MoleculeNet experiments')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['esol', 'freesolv', 'hiv', 'bace', 'clintox', 'tox21'],
                        help='Datasets to run')
    parser.add_argument('--output', type=str, default='moleculenet_extended_results.csv',
                        help='Output filename')
    
    args = parser.parse_args()
    
    print(f"Extended MoleculeNet Experiments - {datetime.now().isoformat()}")
    print(f"Datasets: {args.datasets}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for dataset_name in args.datasets:
        if dataset_name in EXTENDED_DATASETS:
            try:
                results = run_dataset_experiments(dataset_name)
                all_results.extend(results)
            except Exception as e:
                print(f"Error running {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Unknown dataset: {dataset_name}")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = RESULTS_DIR / args.output
        df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to {output_path}")
        print(f"{'='*60}")
        print(df.to_string())
    else:
        print("\nNo results generated.")
    
    return all_results


if __name__ == '__main__':
    main()

