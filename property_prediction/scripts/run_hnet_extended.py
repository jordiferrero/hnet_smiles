#!/usr/bin/env python3
"""
Run H-Net on extended MoleculeNet datasets.
Uses the working featurizer that successfully processed Lipophilicity and BBBP.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score
import xgboost as xgb

from featurizers.hnet_featurizer import HNetFeaturizer, MOLECULE_CHECKPOINTS

CHECKPOINTS_DIR = Path('/home/ec2-user/hnet_smiles/checkpoints')
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'tables'
DATA_DIR = Path(__file__).parent.parent / 'datasets' / 'moleculenet'


def load_esol():
    """Load ESOL (aqueous solubility) dataset."""
    df = pd.read_csv(DATA_DIR / 'esol.csv')
    return df['smiles'].tolist(), df['measured log solubility in mols per litre'].values, 'regression'


def load_freesolv():
    """Load FreeSolv (hydration free energy) dataset."""
    df = pd.read_csv(DATA_DIR / 'freesolv.csv')
    return df['smiles'].tolist(), df['expt'].values, 'regression'


def load_hiv():
    """Load HIV dataset."""
    df = pd.read_csv(DATA_DIR / 'HIV.csv')
    # Remove rows with missing SMILES or target
    df = df.dropna(subset=['smiles', 'HIV_active'])
    return df['smiles'].tolist(), df['HIV_active'].values.astype(int), 'classification'


def load_bace():
    """Load BACE dataset."""
    df = pd.read_csv(DATA_DIR / 'bace.csv')
    return df['mol'].tolist(), df['Class'].values.astype(int), 'classification'


def run_experiments():
    """Run H-Net on all extended datasets."""
    
    # Dataset loaders
    datasets = {
        'ESOL': load_esol,
        'FreeSolv': load_freesolv,
        'HIV': load_hiv,
        'BACE': load_bace,
    }
    
    all_results = []
    
    # Use the best performing molecule checkpoint (from Lipophilicity/BBBP experiments)
    ckpt_name = 'hnet_5ep_nocat_1stg'
    ckpt_run_dir = MOLECULE_CHECKPOINTS[ckpt_name]
    ckpt_path = CHECKPOINTS_DIR / ckpt_run_dir
    
    print(f"Loading H-Net model: {ckpt_name}")
    hnet = HNetFeaturizer(checkpoint_dir=str(ckpt_path), pooling='mean')
    
    for dataset_name, loader in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print('='*60)
        
        try:
            smiles_list, targets, task_type = loader()
            print(f"Loaded {len(smiles_list)} samples, task: {task_type}")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Extract H-Net features
        print(f"Extracting H-Net features...")
        features = []
        valid_idx = []
        
        for i, smi in enumerate(tqdm(smiles_list, desc="H-Net")):
            try:
                feat = hnet.featurize_single(smi)
                if not np.isnan(feat).any():
                    features.append(feat)
                    valid_idx.append(i)
            except Exception as e:
                pass  # Skip failed samples silently
        
        X = np.array(features)
        y = targets[valid_idx]
        
        print(f"Successfully processed {len(valid_idx)}/{len(smiles_list)} samples")
        
        if len(valid_idx) < 100:
            print(f"Too few valid samples, skipping {dataset_name}")
            continue
        
        # Run cross-validation
        n_folds = 5
        
        if task_type == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            fold_aucs = []
            fold_accs = []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                fold_accs.append(accuracy_score(y_test, y_pred))
                try:
                    fold_aucs.append(roc_auc_score(y_test, y_prob))
                except:
                    fold_aucs.append(0.5)
            
            result = {
                'dataset': dataset_name,
                'task': task_type,
                'model': 'H-Net (mean)',
                'accuracy': np.mean(fold_accs),
                'accuracy_std': np.std(fold_accs),
                'auc': np.mean(fold_aucs),
                'auc_std': np.std(fold_aucs),
                'n_samples': len(valid_idx),
                'fold_scores': fold_aucs
            }
            print(f"  AUC: {result['auc']:.4f} ± {result['auc_std']:.4f}")
            
        else:  # regression
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            fold_maes = []
            fold_rmses = []
            fold_r2s = []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                fold_maes.append(mean_absolute_error(y_test, y_pred))
                fold_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                fold_r2s.append(r2_score(y_test, y_pred))
            
            result = {
                'dataset': dataset_name,
                'task': task_type,
                'model': 'H-Net (mean)',
                'mae': np.mean(fold_maes),
                'mae_std': np.std(fold_maes),
                'rmse': np.mean(fold_rmses),
                'rmse_std': np.std(fold_rmses),
                'r2': np.mean(fold_r2s),
                'r2_std': np.std(fold_r2s),
                'n_samples': len(valid_idx),
                'fold_scores': fold_maes
            }
            print(f"  RMSE: {result['rmse']:.4f} ± {result['rmse_std']:.4f}")
            print(f"  MAE: {result['mae']:.4f} ± {result['mae_std']:.4f}")
        
        all_results.append(result)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = RESULTS_DIR / 'hnet_extended_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to {output_path}")
    print(results_df.to_string())
    
    return results_df


if __name__ == '__main__':
    run_experiments()
