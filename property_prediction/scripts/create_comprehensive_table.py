#!/usr/bin/env python3
"""
Create Comprehensive Property Prediction Results Table.

Merges all existing results (polymer + molecular) into a single comprehensive table
and generates publication-ready figures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'tables'
FIGURES_DIR = Path(__file__).parent.parent.parent / 'publication_latex' / 'figures'


def load_all_results() -> pd.DataFrame:
    """Load and consolidate all property prediction results."""
    results = []
    
    # Load existing molecular H-Net results (BBBP, Lipophilicity)
    bbbp_path = RESULTS_DIR / 'bbbp_results.csv'
    if bbbp_path.exists():
        df = pd.read_csv(bbbp_path)
        df['Dataset'] = 'BBBP'
        df['Task'] = 'Classification'
        df['Domain'] = 'Molecular'
        results.append(df)
    
    lipo_path = RESULTS_DIR / 'lipophilicity_results.csv'
    if lipo_path.exists():
        df = pd.read_csv(lipo_path)
        df['Dataset'] = 'Lipophilicity'
        df['Task'] = 'Regression'
        df['Domain'] = 'Molecular'
        results.append(df)
    
    # Load polymer results (Tg, MAC)
    tg_path = RESULTS_DIR / 'tg_results.csv'
    if tg_path.exists():
        df = pd.read_csv(tg_path)
        df['Dataset'] = 'Tg'
        df['Task'] = 'Regression'
        df['Domain'] = 'Polymer'
        results.append(df)
    
    mac_path = RESULTS_DIR / 'mac_results.csv'
    if mac_path.exists():
        df = pd.read_csv(mac_path)
        df['Dataset'] = 'MAC'
        df['Task'] = 'Regression'
        df['Domain'] = 'Polymer'
        results.append(df)
    
    # Load extended MoleculeNet results (RDKit only, for new datasets)
    extended_path = RESULTS_DIR / 'moleculenet_extended_results.csv'
    if extended_path.exists():
        df = pd.read_csv(extended_path)
        # Standardize column names
        if 'dataset' in df.columns:
            df['Dataset'] = df['dataset'].str.upper()
        if 'task' in df.columns:
            df['Task'] = df['task'].str.capitalize()
        if 'model' in df.columns:
            df['Model'] = df['model']
        df['Domain'] = 'Molecular'
        results.append(df)
    
    if results:
        combined = pd.concat(results, ignore_index=True)
        return combined
    
    return pd.DataFrame()


def create_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table for publication."""
    summary_rows = []
    
    # Define datasets in order
    datasets = [
        # Molecular
        ('BBBP', 'Molecular', 'Classification', 'AUC'),
        ('HIV', 'Molecular', 'Classification', 'AUC'),
        ('BACE', 'Molecular', 'Classification', 'AUC'),
        ('CLINTOX', 'Molecular', 'Classification', 'AUC'),
        ('TOX21', 'Molecular', 'Classification', 'AUC'),
        ('Lipophilicity', 'Molecular', 'Regression', 'MAE'),
        ('ESOL', 'Molecular', 'Regression', 'RMSE'),
        ('FREESOLV', 'Molecular', 'Regression', 'RMSE'),
        # Polymer  
        ('Tg', 'Polymer', 'Regression', 'MAE'),
        ('MAC', 'Polymer', 'Regression', 'MAE'),
    ]
    
    for dataset_name, domain, task, metric in datasets:
        # Filter for this dataset
        dataset_results = results[results['Dataset'].str.upper() == dataset_name.upper()]
        
        if dataset_results.empty:
            continue
        
        row = {
            'Dataset': dataset_name,
            'Domain': domain,
            'Task': task,
            'Metric': metric,
        }
        
        # Get RDKit result - check both Model and model columns
        model_col = 'Model' if 'Model' in dataset_results.columns else 'model'
        rdkit_mask = dataset_results[model_col].str.contains('RDKit', case=False, na=False)
        rdkit_results = dataset_results[rdkit_mask]
        
        if not rdkit_results.empty:
            r = rdkit_results.iloc[0]
            val = None
            if metric == 'AUC':
                # Check both uppercase and lowercase column names
                for col in ['AUC', 'auc']:
                    if col in r.index and not pd.isna(r[col]):
                        val = r[col]
                        break
                if val is not None:
                    row['RDKit'] = f"{val:.3f}"
                else:
                    row['RDKit'] = '-'
            elif metric == 'MAE':
                for col in ['MAE', 'mae']:
                    if col in r.index and not pd.isna(r[col]):
                        val = r[col]
                        break
                if val is not None:
                    row['RDKit'] = f"{val:.4f}"
                else:
                    row['RDKit'] = '-'
            elif metric == 'RMSE':
                for col in ['RMSE', 'rmse']:
                    if col in r.index and not pd.isna(r[col]):
                        val = r[col]
                        break
                if val is not None:
                    row['RDKit'] = f"{val:.4f}"
                else:
                    row['RDKit'] = '-'
            else:
                row['RDKit'] = '-'
        else:
            row['RDKit'] = '-'
        
        # Get H-Net result (best among variants)
        hnet_mask = (
            dataset_results[model_col].str.contains('hnet', case=False, na=False) &
            ~dataset_results[model_col].str.contains('cls', case=False, na=False)  # Exclude CLS pooling
        )
        hnet_results = dataset_results[hnet_mask]
        
        if not hnet_results.empty:
            if metric == 'AUC':
                col = 'AUC' if 'AUC' in hnet_results.columns else 'auc'
                if col in hnet_results.columns:
                    best_val = hnet_results[col].max()
                    row['H-Net'] = f"{best_val:.3f}"
            elif metric in ['MAE', 'RMSE']:
                col = metric if metric in hnet_results.columns else metric.lower()
                if col in hnet_results.columns:
                    best_val = hnet_results[col].min()
                    row['H-Net'] = f"{best_val:.4f}"
        else:
            row['H-Net'] = '-'
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def generate_figure(results: pd.DataFrame):
    """Generate publication figure for property prediction results."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Classification tasks (left)
    ax1 = axes[0]
    clf_datasets = ['BBBP', 'HIV', 'BACE', 'CLINTOX', 'TOX21']
    
    rdkit_aucs = []
    hnet_aucs = []
    
    for dataset in clf_datasets:
        dataset_results = results[results['Dataset'].str.upper() == dataset.upper()]
        
        # RDKit
        rdkit = dataset_results[
            dataset_results['Model'].str.contains('RDKit', case=False, na=False) |
            dataset_results.get('model', pd.Series([''])).str.contains('RDKit', case=False, na=False)
        ]
        if not rdkit.empty:
            auc_col = 'AUC' if 'AUC' in rdkit.columns else 'auc'
            rdkit_aucs.append(rdkit[auc_col].values[0] if auc_col in rdkit.columns else 0)
        else:
            rdkit_aucs.append(0)
        
        # H-Net
        hnet = dataset_results[
            (dataset_results['Model'].str.contains('hnet', case=False, na=False) |
             dataset_results.get('model', pd.Series([''])).str.contains('hnet', case=False, na=False)) &
            ~dataset_results['Model'].str.contains('cls', case=False, na=False)
        ]
        if not hnet.empty:
            auc_col = 'AUC' if 'AUC' in hnet.columns else 'auc'
            hnet_aucs.append(hnet[auc_col].max() if auc_col in hnet.columns else 0)
        else:
            hnet_aucs.append(0)
    
    x = np.arange(len(clf_datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, rdkit_aucs, width, label='RDKit', color='#e67e22')
    bars2 = ax1.bar(x + width/2, hnet_aucs, width, label='H-Net', color='#3498db')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('AUC-ROC')
    ax1.set_title('Classification Tasks')
    ax1.set_xticks(x)
    ax1.set_xticklabels(clf_datasets, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0.5, 1.0)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Regression tasks (right)
    ax2 = axes[1]
    reg_datasets = ['Lipophilicity', 'ESOL', 'FREESOLV', 'Tg']
    
    rdkit_errors = []
    hnet_errors = []
    
    for dataset in reg_datasets:
        dataset_results = results[results['Dataset'].str.upper() == dataset.upper()]
        
        # RDKit
        rdkit = dataset_results[
            dataset_results['Model'].str.contains('RDKit', case=False, na=False) |
            dataset_results.get('model', pd.Series([''])).str.contains('RDKit', case=False, na=False)
        ]
        if not rdkit.empty:
            if 'MAE' in rdkit.columns:
                rdkit_errors.append(rdkit['MAE'].values[0])
            elif 'mae' in rdkit.columns:
                rdkit_errors.append(rdkit['mae'].values[0])
            elif 'rmse' in rdkit.columns:
                rdkit_errors.append(rdkit['rmse'].values[0])
            else:
                rdkit_errors.append(0)
        else:
            rdkit_errors.append(0)
        
        # H-Net
        hnet = dataset_results[
            (dataset_results['Model'].str.contains('hnet', case=False, na=False) |
             dataset_results.get('model', pd.Series([''])).str.contains('hnet', case=False, na=False)) &
            ~dataset_results['Model'].str.contains('cls', case=False, na=False)
        ]
        if not hnet.empty:
            if 'MAE' in hnet.columns:
                hnet_errors.append(hnet['MAE'].min())
            elif 'mae' in hnet.columns:
                hnet_errors.append(hnet['mae'].min())
            elif 'rmse' in hnet.columns:
                hnet_errors.append(hnet['rmse'].min())
            else:
                hnet_errors.append(0)
        else:
            hnet_errors.append(0)
    
    x = np.arange(len(reg_datasets))
    
    bars1 = ax2.bar(x - width/2, rdkit_errors, width, label='RDKit', color='#e67e22')
    bars2 = ax2.bar(x + width/2, hnet_errors, width, label='H-Net', color='#3498db')
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Error (lower is better)')
    ax2.set_title('Regression Tasks')
    ax2.set_xticks(x)
    ax2.set_xticklabels(reg_datasets, rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save
    pdf_path = FIGURES_DIR / 'property_prediction_extended.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved to {pdf_path}")
    
    png_path = FIGURES_DIR / 'property_prediction_extended.png'
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved to {png_path}")
    
    plt.close()


def main():
    """Main function."""
    print("Loading all results...")
    results = load_all_results()
    
    print(f"Total rows: {len(results)}")
    print(f"Unique datasets: {results['Dataset'].unique().tolist()}")
    
    # Create summary table
    print("\nCreating summary table...")
    summary = create_summary_table(results)
    print(summary.to_string())
    
    # Save
    summary_path = RESULTS_DIR / 'comprehensive_results.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved to {summary_path}")
    
    # Generate figure
    print("\nGenerating figure...")
    generate_figure(results)
    
    return summary


if __name__ == '__main__':
    summary = main()

