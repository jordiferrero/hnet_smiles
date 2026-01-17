#!/usr/bin/env python3
"""
Full property prediction experiments.

Compares H-Net models with RDKit baselines on:
- MOLECULES: Lipophilicity (regression), BBBP (classification)
- POLYMERS: Tg (regression), MAC (regression)

Uses 10,000 samples per dataset maximum and 5-fold CV.
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
import json
import argparse

from datasets import load_lipophilicity, load_bbbp, load_pi1m_tg_mac
from featurizers import RDKitFeaturizer
from featurizers.hnet_featurizer import HNetFeaturizer, MOLECULE_CHECKPOINTS, POLYMER_CHECKPOINTS
from models.xgboost_predictor import XGBoostPredictor


CHECKPOINTS_DIR = Path('/home/ec2-user/hnet_smiles/checkpoints')
RESULTS_DIR = Path(__file__).parent.parent / 'results'


def run_molecule_experiments(max_samples=10000, n_folds=5, pooling_strategies=['mean', 'cls']):
    """Run experiments on molecule datasets."""
    print("\n" + "="*70)
    print("MOLECULE EXPERIMENTS")
    print("="*70)
    
    results = []
    
    # Load datasets
    print(f"\nLoading datasets (max {max_samples} samples)...")
    lipo = load_lipophilicity(max_samples=max_samples)
    bbbp = load_bbbp(max_samples=max_samples)
    
    # ===== RDKit Baseline =====
    print("\n" + "-"*50)
    print("RDKit Baseline")
    print("-"*50)
    
    rdkit = RDKitFeaturizer(feature_type='combined', handle_polymer_smiles=False)
    
    # Lipophilicity
    print("\nLipophilicity (regression):")
    X_lipo = rdkit.featurize_batch(lipo.smiles.tolist(), show_progress=True)
    predictor = XGBoostPredictor(task_type='regression', n_folds=n_folds)
    scores = predictor.train_cv(X_lipo, lipo.targets, verbose=True)
    results.append({
        'Model': 'RDKit', 'Pooling': '-', 'Dataset': 'Lipophilicity',
        'Task': 'regression', 'MAE': scores['mae'], 'MAE_std': scores['mae_std'],
        'RMSE': scores['rmse'], 'R2': scores['r2']
    })
    
    # BBBP
    print("\nBBBP (classification):")
    X_bbbp = rdkit.featurize_batch(bbbp.smiles.tolist(), show_progress=True)
    predictor = XGBoostPredictor(task_type='classification', n_folds=n_folds)
    scores = predictor.train_cv(X_bbbp, bbbp.targets, verbose=True)
    results.append({
        'Model': 'RDKit', 'Pooling': '-', 'Dataset': 'BBBP',
        'Task': 'classification', 'Accuracy': scores['accuracy'], 
        'Accuracy_std': scores['accuracy_std'], 'AUC': scores['auc'], 'AUC_std': scores['auc_std']
    })
    
    # ===== H-Net Models =====
    for ckpt_name, ckpt_run_dir in MOLECULE_CHECKPOINTS.items():
        ckpt_path = CHECKPOINTS_DIR / ckpt_run_dir
        
        if not ckpt_path.exists():
            print(f"\nSkipping {ckpt_name} (checkpoint not found)")
            continue
        
        for pooling in pooling_strategies:
            print("\n" + "-"*50)
            print(f"H-Net: {ckpt_name} (pooling: {pooling})")
            print("-"*50)
            
            try:
                hnet = HNetFeaturizer(str(ckpt_path), pooling=pooling)
                
                # Lipophilicity
                print("\nLipophilicity (regression):")
                X_lipo_hnet = hnet.featurize_batch(lipo.smiles.tolist(), show_progress=True)
                predictor = XGBoostPredictor(task_type='regression', n_folds=n_folds)
                scores = predictor.train_cv(X_lipo_hnet, lipo.targets, verbose=True)
                results.append({
                    'Model': ckpt_name, 'Pooling': pooling, 'Dataset': 'Lipophilicity',
                    'Task': 'regression', 'MAE': scores['mae'], 'MAE_std': scores['mae_std'],
                    'RMSE': scores['rmse'], 'R2': scores['r2']
                })
                
                # BBBP
                print("\nBBBP (classification):")
                X_bbbp_hnet = hnet.featurize_batch(bbbp.smiles.tolist(), show_progress=True)
                predictor = XGBoostPredictor(task_type='classification', n_folds=n_folds)
                scores = predictor.train_cv(X_bbbp_hnet, bbbp.targets, verbose=True)
                results.append({
                    'Model': ckpt_name, 'Pooling': pooling, 'Dataset': 'BBBP',
                    'Task': 'classification', 'Accuracy': scores['accuracy'],
                    'Accuracy_std': scores['accuracy_std'], 'AUC': scores['auc'], 'AUC_std': scores['auc_std']
                })
                
            except Exception as e:
                print(f"Error with {ckpt_name}/{pooling}: {e}")
    
    return results


def run_polymer_experiments(max_samples=10000, n_folds=5, pooling_strategies=['mean', 'cls']):
    """Run experiments on polymer datasets."""
    print("\n" + "="*70)
    print("POLYMER EXPERIMENTS")
    print("="*70)
    
    results = []
    
    # Load datasets
    print(f"\nLoading datasets (max {max_samples} samples)...")
    tg_data = load_pi1m_tg_mac(target='Tg', max_samples=max_samples)
    mac_data = load_pi1m_tg_mac(target='MAC', max_samples=max_samples)
    
    # ===== RDKit Baseline =====
    print("\n" + "-"*50)
    print("RDKit Baseline (polymer mode)")
    print("-"*50)
    
    rdkit = RDKitFeaturizer(feature_type='combined', handle_polymer_smiles=True)
    
    # Tg
    print("\nTg (regression):")
    X_tg = rdkit.featurize_batch(tg_data.smiles.tolist(), show_progress=True)
    predictor = XGBoostPredictor(task_type='regression', n_folds=n_folds)
    scores = predictor.train_cv(X_tg, tg_data.targets, verbose=True)
    results.append({
        'Model': 'RDKit', 'Pooling': '-', 'Dataset': 'Tg',
        'Task': 'regression', 'MAE': scores['mae'], 'MAE_std': scores['mae_std'],
        'RMSE': scores['rmse'], 'R2': scores['r2']
    })
    
    # MAC
    print("\nMAC (regression):")
    X_mac = rdkit.featurize_batch(mac_data.smiles.tolist(), show_progress=True)
    predictor = XGBoostPredictor(task_type='regression', n_folds=n_folds)
    scores = predictor.train_cv(X_mac, mac_data.targets, verbose=True)
    results.append({
        'Model': 'RDKit', 'Pooling': '-', 'Dataset': 'MAC',
        'Task': 'regression', 'MAE': scores['mae'], 'MAE_std': scores['mae_std'],
        'RMSE': scores['rmse'], 'R2': scores['r2']
    })
    
    # ===== H-Net Models =====
    for ckpt_name, ckpt_run_dir in POLYMER_CHECKPOINTS.items():
        ckpt_path = CHECKPOINTS_DIR / ckpt_run_dir
        
        if not ckpt_path.exists():
            print(f"\nSkipping {ckpt_name} (checkpoint not found)")
            continue
        
        for pooling in pooling_strategies:
            print("\n" + "-"*50)
            print(f"H-Net: {ckpt_name} (pooling: {pooling})")
            print("-"*50)
            
            try:
                hnet = HNetFeaturizer(str(ckpt_path), pooling=pooling)
                
                # Tg
                print("\nTg (regression):")
                X_tg_hnet = hnet.featurize_batch(tg_data.smiles.tolist(), show_progress=True)
                predictor = XGBoostPredictor(task_type='regression', n_folds=n_folds)
                scores = predictor.train_cv(X_tg_hnet, tg_data.targets, verbose=True)
                results.append({
                    'Model': ckpt_name, 'Pooling': pooling, 'Dataset': 'Tg',
                    'Task': 'regression', 'MAE': scores['mae'], 'MAE_std': scores['mae_std'],
                    'RMSE': scores['rmse'], 'R2': scores['r2']
                })
                
                # MAC
                print("\nMAC (regression):")
                X_mac_hnet = hnet.featurize_batch(mac_data.smiles.tolist(), show_progress=True)
                predictor = XGBoostPredictor(task_type='regression', n_folds=n_folds)
                scores = predictor.train_cv(X_mac_hnet, mac_data.targets, verbose=True)
                results.append({
                    'Model': ckpt_name, 'Pooling': pooling, 'Dataset': 'MAC',
                    'Task': 'regression', 'MAE': scores['mae'], 'MAE_std': scores['mae_std'],
                    'RMSE': scores['rmse'], 'R2': scores['r2']
                })
                
            except Exception as e:
                print(f"Error with {ckpt_name}/{pooling}: {e}")
    
    return results


def create_summary_tables(molecule_results, polymer_results):
    """Create formatted summary tables."""
    tables_dir = RESULTS_DIR / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    all_results = molecule_results + polymer_results
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(tables_dir / 'all_results.csv', index=False)
    
    # ===== Molecule Summary =====
    if molecule_results:
        print("\n" + "="*70)
        print("MOLECULE RESULTS SUMMARY")
        print("="*70)
        
        # Lipophilicity table
        lipo_results = [r for r in molecule_results if r['Dataset'] == 'Lipophilicity']
        if lipo_results:
            df_lipo = pd.DataFrame(lipo_results)[['Model', 'Pooling', 'MAE', 'MAE_std', 'RMSE', 'R2']]
            df_lipo['MAE±std'] = df_lipo.apply(lambda r: f"{r['MAE']:.4f}±{r['MAE_std']:.4f}", axis=1)
            df_lipo = df_lipo.sort_values('MAE')
            
            print("\nLipophilicity (MAE ↓ is better):")
            print(df_lipo[['Model', 'Pooling', 'MAE±std', 'RMSE', 'R2']].to_string(index=False))
            df_lipo.to_csv(tables_dir / 'lipophilicity_results.csv', index=False)
        
        # BBBP table
        bbbp_results = [r for r in molecule_results if r['Dataset'] == 'BBBP']
        if bbbp_results:
            df_bbbp = pd.DataFrame(bbbp_results)[['Model', 'Pooling', 'Accuracy', 'Accuracy_std', 'AUC', 'AUC_std']]
            df_bbbp['Acc±std'] = df_bbbp.apply(lambda r: f"{r['Accuracy']:.4f}±{r['Accuracy_std']:.4f}", axis=1)
            df_bbbp['AUC±std'] = df_bbbp.apply(lambda r: f"{r['AUC']:.4f}±{r['AUC_std']:.4f}", axis=1)
            df_bbbp = df_bbbp.sort_values('AUC', ascending=False)
            
            print("\nBBBP (AUC ↑ is better):")
            print(df_bbbp[['Model', 'Pooling', 'Acc±std', 'AUC±std']].to_string(index=False))
            df_bbbp.to_csv(tables_dir / 'bbbp_results.csv', index=False)
    
    # ===== Polymer Summary =====
    if polymer_results:
        print("\n" + "="*70)
        print("POLYMER RESULTS SUMMARY")
        print("="*70)
        
        # Tg table
        tg_results = [r for r in polymer_results if r['Dataset'] == 'Tg']
        if tg_results:
            df_tg = pd.DataFrame(tg_results)[['Model', 'Pooling', 'MAE', 'MAE_std', 'RMSE', 'R2']]
            df_tg['MAE±std'] = df_tg.apply(lambda r: f"{r['MAE']:.4f}±{r['MAE_std']:.4f}", axis=1)
            df_tg = df_tg.sort_values('MAE')
            
            print("\nTg (MAE ↓ is better):")
            print(df_tg[['Model', 'Pooling', 'MAE±std', 'RMSE', 'R2']].to_string(index=False))
            df_tg.to_csv(tables_dir / 'tg_results.csv', index=False)
        
        # MAC table
        mac_results = [r for r in polymer_results if r['Dataset'] == 'MAC']
        if mac_results:
            df_mac = pd.DataFrame(mac_results)[['Model', 'Pooling', 'MAE', 'MAE_std', 'RMSE', 'R2']]
            df_mac['MAE±std'] = df_mac.apply(lambda r: f"{r['MAE']:.6f}±{r['MAE_std']:.6f}", axis=1)
            df_mac = df_mac.sort_values('MAE')
            
            print("\nMAC (MAE ↓ is better):")
            print(df_mac[['Model', 'Pooling', 'MAE±std', 'RMSE', 'R2']].to_string(index=False))
            df_mac.to_csv(tables_dir / 'mac_results.csv', index=False)
    
    return tables_dir


def main():
    parser = argparse.ArgumentParser(description='Run property prediction experiments')
    parser.add_argument('--dataset', choices=['molecule', 'polymer', 'all'], default='all',
                        help='Which dataset group to run')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Maximum samples per dataset (default: 10000)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--pooling', nargs='+', default=['mean', 'cls'],
                        help='Pooling strategies to test (default: mean cls)')
    args = parser.parse_args()
    
    print("="*70)
    print("PROPERTY PREDICTION EXPERIMENTS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Max samples: {args.max_samples}")
    print(f"CV folds: {args.n_folds}")
    print(f"Pooling strategies: {args.pooling}")
    print("="*70)
    
    molecule_results = []
    polymer_results = []
    
    if args.dataset in ['molecule', 'all']:
        molecule_results = run_molecule_experiments(
            max_samples=args.max_samples,
            n_folds=args.n_folds,
            pooling_strategies=args.pooling
        )
    
    if args.dataset in ['polymer', 'all']:
        polymer_results = run_polymer_experiments(
            max_samples=args.max_samples,
            n_folds=args.n_folds,
            pooling_strategies=args.pooling
        )
    
    # Create summary tables
    if molecule_results or polymer_results:
        tables_dir = create_summary_tables(molecule_results, polymer_results)
        print(f"\n✓ Results saved to: {tables_dir}")
    
    print("\n✓ Experiments completed!")


if __name__ == '__main__':
    main()

