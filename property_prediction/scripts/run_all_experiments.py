#!/usr/bin/env python3
"""
Run all property prediction experiments.

Uses cached features from extract_all_features.py and trains XGBoost models.

Usage:
    python scripts/run_all_experiments.py --dataset polymer
    python scripts/run_all_experiments.py --dataset molecule
    python scripts/run_all_experiments.py --dataset all
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import XGBoostPredictor
from sklearn.model_selection import train_test_split


# Paths
FEATURES_DIR = Path(__file__).parent.parent / 'results' / 'features'
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'tables'


def load_cached_features(dataset_name: str, featurizer_name: str) -> dict:
    """Load cached features from disk."""
    feature_path = FEATURES_DIR / dataset_name / f'{featurizer_name}.npz'
    
    if not feature_path.exists():
        return None
    
    data = np.load(feature_path)
    return {
        'features': data['features'],
        'feature_names': data['feature_names'].tolist() if 'feature_names' in data else None,
    }


def get_available_features(dataset_name: str) -> list:
    """List available feature files for a dataset."""
    feature_dir = FEATURES_DIR / dataset_name
    
    if not feature_dir.exists():
        return []
    
    return [f.stem for f in feature_dir.glob('*.npz')]


def run_polymer_experiments() -> pd.DataFrame:
    """Run all polymer property prediction experiments."""
    print("\n" + "=" * 60)
    print("Running POLYMER experiments (Tg, MAC)")
    print("=" * 60)
    
    # Check for targets
    targets_dir = FEATURES_DIR / 'polymer'
    if not targets_dir.exists():
        print("No polymer features found. Run extract_all_features.py first.")
        return pd.DataFrame()
    
    # Load targets
    targets_tg = np.load(targets_dir / 'targets_Tg.npy')
    targets_mac = np.load(targets_dir / 'targets_MAC.npy')
    
    # Get available features
    available = get_available_features('polymer')
    print(f"Available features: {available}")
    
    results = []
    
    for target_name, targets in [('Tg', targets_tg), ('MAC', targets_mac)]:
        print(f"\n--- {target_name} Prediction ---")
        
        for feat_name in available:
            if feat_name in ['smiles', 'targets_Tg', 'targets_MAC']:
                continue
            
            print(f"\n  Featurizer: {feat_name}")
            
            data = load_cached_features('polymer', feat_name)
            if data is None:
                print(f"    Skipping: features not found")
                continue
            
            features = data['features']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets,
                test_size=0.1,
                random_state=42,
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.111,  # 0.1 of original
                random_state=42,
            )
            
            print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Train XGBoost
            predictor = XGBoostPredictor(task_type='regression')
            
            try:
                cv_scores = predictor.train_cv(X_train, y_train, verbose=False)
                
                # Test evaluation
                y_pred = predictor.predict(X_test)
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                test_mae = mean_absolute_error(y_test, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_r2 = r2_score(y_test, y_pred)
                
                print(f"    CV MAE: {cv_scores['mae']:.4f} ± {cv_scores['mae_std']:.4f}")
                print(f"    Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
                
                results.append({
                    'dataset': 'PI1M_Tg_MAC',
                    'target': target_name,
                    'featurizer': feat_name,
                    'cv_mae': cv_scores['mae'],
                    'cv_mae_std': cv_scores['mae_std'],
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2,
                })
            
            except Exception as e:
                print(f"    Error: {e}")
    
    return pd.DataFrame(results)


def run_molecule_experiments() -> pd.DataFrame:
    """Run all molecule property prediction experiments."""
    print("\n" + "=" * 60)
    print("Running MOLECULE experiments (Lipophilicity, BBBP)")
    print("=" * 60)
    
    results = []
    
    # Lipophilicity (regression)
    print("\n--- Lipophilicity Prediction ---")
    targets_dir = FEATURES_DIR / 'lipophilicity'
    
    if targets_dir.exists() and (targets_dir / 'targets.npy').exists():
        targets = np.load(targets_dir / 'targets.npy')
        available = get_available_features('lipophilicity')
        
        for feat_name in available:
            if feat_name in ['smiles', 'targets']:
                continue
            
            print(f"\n  Featurizer: {feat_name}")
            
            data = load_cached_features('lipophilicity', feat_name)
            if data is None:
                continue
            
            features = data['features']
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.1, random_state=42
            )
            
            predictor = XGBoostPredictor(task_type='regression')
            
            try:
                cv_scores = predictor.train_cv(X_train, y_train, verbose=False)
                y_pred = predictor.predict(X_test)
                
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                test_mae = mean_absolute_error(y_test, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_r2 = r2_score(y_test, y_pred)
                
                print(f"    CV MAE: {cv_scores['mae']:.4f}, Test MAE: {test_mae:.4f}")
                
                results.append({
                    'dataset': 'Lipophilicity',
                    'target': 'logD',
                    'featurizer': feat_name,
                    'cv_mae': cv_scores['mae'],
                    'cv_mae_std': cv_scores['mae_std'],
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2,
                })
            except Exception as e:
                print(f"    Error: {e}")
    else:
        print("  Lipophilicity features not found")
    
    # BBBP (classification)
    print("\n--- BBBP Prediction ---")
    targets_dir = FEATURES_DIR / 'bbbp'
    
    if targets_dir.exists() and (targets_dir / 'targets.npy').exists():
        targets = np.load(targets_dir / 'targets.npy')
        available = get_available_features('bbbp')
        
        for feat_name in available:
            if feat_name in ['smiles', 'targets']:
                continue
            
            print(f"\n  Featurizer: {feat_name}")
            
            data = load_cached_features('bbbp', feat_name)
            if data is None:
                continue
            
            features = data['features']
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.1, random_state=42, stratify=targets
            )
            
            predictor = XGBoostPredictor(task_type='classification')
            
            try:
                cv_scores = predictor.train_cv(X_train, y_train, verbose=False)
                y_pred = predictor.predict(X_test)
                y_prob = predictor.predict_proba(X_test)
                
                from sklearn.metrics import accuracy_score, roc_auc_score
                test_acc = accuracy_score(y_test, y_pred)
                test_auc = roc_auc_score(y_test, y_prob)
                
                print(f"    CV Acc: {cv_scores['accuracy']:.4f}, Test Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
                
                results.append({
                    'dataset': 'BBBP',
                    'target': 'penetration',
                    'featurizer': feat_name,
                    'cv_accuracy': cv_scores['accuracy'],
                    'cv_auc': cv_scores['auc'],
                    'test_accuracy': test_acc,
                    'test_auc': test_auc,
                })
            except Exception as e:
                print(f"    Error: {e}")
    else:
        print("  BBBP features not found")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Run property prediction experiments')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['polymer', 'molecule', 'all'],
        default='all',
        help='Which experiments to run',
    )
    
    args = parser.parse_args()
    
    print(f"Property Prediction Experiments - {datetime.now().isoformat()}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    if args.dataset in ['polymer', 'all']:
        polymer_results = run_polymer_experiments()
        if not polymer_results.empty:
            polymer_results.to_csv(RESULTS_DIR / 'polymer_results.csv', index=False)
            all_results.append(polymer_results)
    
    if args.dataset in ['molecule', 'all']:
        molecule_results = run_molecule_experiments()
        if not molecule_results.empty:
            molecule_results.to_csv(RESULTS_DIR / 'molecule_results.csv', index=False)
            all_results.append(molecule_results)
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(RESULTS_DIR / 'all_results.csv', index=False)
        
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(combined.to_string())
        print(f"\nResults saved to: {RESULTS_DIR}")
    else:
        print("\nNo results generated. Ensure features are extracted first.")


if __name__ == '__main__':
    main()











