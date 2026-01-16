#!/usr/bin/env python3
"""
Compression Metrics Analysis for H-Net on Chemical SMILES

This script analyzes Bits-Per-Byte (BPB) and Perplexity (PPL) metrics
to evaluate H-Net's compression efficiency on chemical data.

Based on the H-Net paper (Hwang et al., 2025):
- BPB = CE_loss / ln(2) - measures bits needed per byte
- PPL = exp(CE_loss) - measures "surprise" per token

Usage:
    python compression_metrics_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Setup plotting style
sns.set_theme(style='whitegrid', context='talk', palette='mako')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Project paths
project_root = Path(__file__).parent.parent
analysis_dir = project_root / 'analysis'
checkpoints_dir = project_root / 'checkpoints'
figures_dir = analysis_dir / 'figures'
data_dir = analysis_dir / 'data'

figures_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Core Functions: BPB and Perplexity Calculations
# =============================================================================

def ce_to_bpb(ce_loss: float) -> float:
    """
    Convert Cross-Entropy loss (in nats) to Bits-Per-Byte.
    
    BPB = CE_loss / ln(2)
    
    For byte-level models:
    - Theoretical maximum: 8 bits/byte (log2(256) for random prediction)
    - Well-trained model: ~1.0-2.0 bits/byte
    - Perfect compression: ~0 bits/byte (impossible in practice)
    """
    return ce_loss / np.log(2)


def ce_to_perplexity(ce_loss: float) -> float:
    """
    Convert Cross-Entropy loss (in nats) to Perplexity.
    
    PPL = exp(CE_loss)
    
    Interpretation:
    - PPL = 1: Perfect prediction (impossible)
    - PPL = 256: Random prediction for byte-level
    - Lower is better
    """
    return np.exp(ce_loss)


def load_training_history(model_path: Path) -> Dict:
    """Load training history from metadata.json."""
    metadata_path = model_path / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def extract_compression_metrics(model_path: Path) -> pd.DataFrame:
    """
    Extract BPB and perplexity from training history.
    
    Returns DataFrame with columns:
    - training_bytes: Cumulative training bytes
    - ce_loss: Cross-entropy loss (nats)
    - bpb: Bits-per-byte
    - perplexity: Perplexity
    """
    metadata = load_training_history(model_path)
    history = metadata.get('training_history', [])
    
    if not history:
        return pd.DataFrame()
    
    records = []
    for entry in history:
        metrics = entry.get('metrics', {})
        ce_loss = metrics.get('ce_loss', None)
        
        if ce_loss is not None:
            records.append({
                'training_bytes': entry.get('cumulative_training_bytes', 0),
                'bytes_threshold': entry.get('bytes_threshold', 0),
                'ce_loss': ce_loss,
                'bpb': ce_to_bpb(ce_loss),
                'perplexity': ce_to_perplexity(ce_loss),
                'checkpoint_type': entry.get('checkpoint_type', 'unknown'),
                'epoch': entry.get('epoch', None),
            })
    
    return pd.DataFrame(records)


def get_final_metrics(model_path: Path) -> Dict:
    """Get final (best) compression metrics for a model."""
    df = extract_compression_metrics(model_path)
    
    if df.empty:
        return {'ce_loss': None, 'bpb': None, 'perplexity': None}
    
    # Get the final checkpoint (highest training bytes)
    final = df.loc[df['training_bytes'].idxmax()]
    
    # Also get the best (lowest) values
    best_ce = df['ce_loss'].min()
    
    return {
        'final_ce_loss': final['ce_loss'],
        'final_bpb': final['bpb'],
        'final_perplexity': final['perplexity'],
        'best_ce_loss': best_ce,
        'best_bpb': ce_to_bpb(best_ce),
        'best_perplexity': ce_to_perplexity(best_ce),
        'total_training_bytes': final['training_bytes'],
    }


# =============================================================================
# Model Definitions
# =============================================================================

MODELS = {
    # 1-Stage Architecture
    'PI1M_noconcat_5epoch': {
        'path': checkpoints_dir / 'run_large_20251111_075600',
        'dataset': 'PI1M',
        'concatenate': False,
        'epochs': 5,
        'architecture': '1-stage',
    },
    'PI1M_concat_1epoch': {
        'path': checkpoints_dir / 'run_large_20251113_181705',
        'dataset': 'PI1M',
        'concatenate': True,
        'epochs': 1,
        'architecture': '1-stage',
    },
    'PI1M_concat_5epoch': {
        'path': checkpoints_dir / 'run_large_20251111_181836',
        'dataset': 'PI1M',
        'concatenate': True,
        'epochs': 5,
        'architecture': '1-stage',
    },
    'PI1M_concat_22epoch': {
        'path': checkpoints_dir / 'run_large_20251112_150502',
        'dataset': 'PI1M',
        'concatenate': True,
        'epochs': 22,
        'architecture': '1-stage',
    },
    'MOSES_noconcat_5epoch': {
        'path': checkpoints_dir / 'run_large_20251113_074900',
        'dataset': 'MOSES',
        'concatenate': False,
        'epochs': 5,
        'architecture': '1-stage',
    },
    'MOSES_concat_5epoch': {
        'path': checkpoints_dir / 'run_large_20251112_071557',
        'dataset': 'MOSES',
        'concatenate': True,
        'epochs': 5,
        'architecture': '1-stage',
    },
    # 2-Stage Architecture (TODO: Update paths when training completes)
    'PI1M_concat_5epoch_2stage': {
        'path': checkpoints_dir / 'run_large_20260116_074355',  # Currently training
        'dataset': 'PI1M',
        'concatenate': True,
        'epochs': 5,
        'architecture': '2-stage',
    },
    'MOSES_concat_5epoch_2stage': {
        'path': checkpoints_dir / 'run_large_2stage_MOSES',
        'dataset': 'MOSES',
        'concatenate': True,
        'epochs': 5,
        'architecture': '2-stage',
    },
}


# =============================================================================
# Analysis Functions
# =============================================================================

def load_all_models():
    """Load compression metrics for all models."""
    all_metrics = {}
    all_histories = {}
    
    print("Loading compression metrics for all models...")
    print("=" * 60)
    
    for model_name, model_info in MODELS.items():
        model_path = model_info['path']
        
        if not model_path.exists():
            print(f"⏳ {model_name}: Path not found (training may be pending)")
            all_metrics[model_name] = None
            continue
        
        try:
            # Get final metrics
            metrics = get_final_metrics(model_path)
            metrics.update(model_info)
            all_metrics[model_name] = metrics
            
            # Get full history for plotting
            history_df = extract_compression_metrics(model_path)
            all_histories[model_name] = history_df
            
            print(f"✓ {model_name}: BPB={metrics['final_bpb']:.3f}, PPL={metrics['final_perplexity']:.2f}")
        except Exception as e:
            print(f"✗ {model_name}: Error - {e}")
            all_metrics[model_name] = None
    
    print("=" * 60)
    print(f"Loaded {sum(1 for v in all_metrics.values() if v is not None)} models.")
    
    return all_metrics, all_histories


def create_summary_table(all_metrics: Dict) -> pd.DataFrame:
    """Create summary table of all models."""
    summary_data = []
    
    for model_name, metrics in all_metrics.items():
        if metrics is None:
            summary_data.append({
                'Model': model_name,
                'Dataset': MODELS[model_name]['dataset'],
                'Concatenation': 'Yes' if MODELS[model_name]['concatenate'] else 'No',
                'Epochs': MODELS[model_name]['epochs'],
                'Architecture': MODELS[model_name]['architecture'],
                'Final BPB': '⏳ Pending',
                'Final PPL': '⏳ Pending',
                'Best BPB': '⏳ Pending',
                'Training Bytes': '⏳ Pending',
            })
        else:
            summary_data.append({
                'Model': model_name,
                'Dataset': metrics['dataset'],
                'Concatenation': 'Yes' if metrics['concatenate'] else 'No',
                'Epochs': metrics['epochs'],
                'Architecture': metrics['architecture'],
                'Final BPB': f"{metrics['final_bpb']:.3f}",
                'Final PPL': f"{metrics['final_perplexity']:.2f}",
                'Best BPB': f"{metrics['best_bpb']:.3f}",
                'Training Bytes': f"{metrics['total_training_bytes']:,}",
            })
    
    return pd.DataFrame(summary_data)


def plot_bpb_training_dynamics(all_histories: Dict):
    """Plot BPB vs Training Bytes for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for model_name, history_df in all_histories.items():
        if history_df is None or history_df.empty:
            continue
        
        model_info = MODELS[model_name]
        ax = axes[0] if model_info['dataset'] == 'PI1M' else axes[1]
        
        # Plot BPB over training
        label = model_name.replace('_', ' ')
        linestyle = '--' if model_info['architecture'] == '2-stage' else '-'
        ax.plot(
            history_df['training_bytes'] / 1e6, 
            history_df['bpb'],
            label=label,
            linestyle=linestyle,
            linewidth=2,
            marker='o' if len(history_df) < 50 else None,
            markersize=4,
        )
    
    # Customize axes
    for ax, title in zip(axes, ['PI1M (Polymer)', 'MOSES (Molecular)']):
        ax.set_xlabel('Training Bytes (M)')
        ax.set_ylabel('Bits-Per-Byte (BPB)')
        ax.set_title(f'{title} - Compression Efficiency Over Training')
        ax.legend(loc='upper right', fontsize=9)
        ax.axhline(y=8, color='red', linestyle=':', alpha=0.5, label='Random (8 BPB)')
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'compression_bpb_training_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {figures_dir / 'compression_bpb_training_dynamics.png'}")


def plot_bpb_comparison(all_metrics: Dict):
    """Create comparison bar chart of final BPB."""
    from matplotlib.patches import Patch
    
    available_metrics = {k: v for k, v in all_metrics.items() 
                        if v is not None and 'final_bpb' in v}
    
    if not available_metrics:
        print("No models with metrics available yet.")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for bar chart
    model_names = list(available_metrics.keys())
    bpb_values = [m['final_bpb'] for m in available_metrics.values()]
    architectures = [MODELS[k]['architecture'] for k in model_names]
    datasets = [MODELS[k]['dataset'] for k in model_names]
    
    # Create color coding by dataset and pattern by architecture
    colors = ['#4e79a7' if d == 'PI1M' else '#59a14f' for d in datasets]
    hatches = ['/' if a == '2-stage' else '' for a in architectures]
    
    bars = ax.bar(range(len(model_names)), bpb_values, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    # Customize
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in model_names], rotation=45, ha='right')
    ax.set_ylabel('Bits-Per-Byte (BPB)')
    ax.set_title('Final Compression Efficiency by Model\n(Lower is Better)')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, bpb_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#4e79a7', label='PI1M (Polymer)'),
        Patch(facecolor='#59a14f', label='MOSES (Molecular)'),
        Patch(facecolor='gray', hatch='/', label='2-Stage Architecture'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add reference line for theoretical maximum
    ax.axhline(y=8, color='red', linestyle=':', alpha=0.3)
    ax.text(len(model_names) - 0.5, 8.1, 'Random (8 BPB)', fontsize=9, color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'compression_bpb_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {figures_dir / 'compression_bpb_comparison.png'}")


def plot_training_amount_effect(all_metrics: Dict):
    """Compare compression across 1, 5, 22 epochs."""
    epoch_models = ['PI1M_concat_1epoch', 'PI1M_concat_5epoch', 'PI1M_concat_22epoch']
    epoch_metrics = {k: all_metrics.get(k) for k in epoch_models 
                    if all_metrics.get(k) is not None}
    
    if not epoch_metrics:
        print("Epoch comparison models not available.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    epochs = [m['epochs'] for m in epoch_metrics.values()]
    bpb_values = [m['final_bpb'] for m in epoch_metrics.values()]
    ppl_values = [m['final_perplexity'] for m in epoch_metrics.values()]
    
    # BPB by epoch
    colors = sns.color_palette('mako', len(epochs))
    axes[0].bar(range(len(epochs)), bpb_values, color=colors)
    axes[0].set_xticks(range(len(epochs)))
    axes[0].set_xticklabels([f'{e} epoch{"s" if e > 1 else ""}' for e in epochs])
    axes[0].set_ylabel('Bits-Per-Byte (BPB)')
    axes[0].set_title('Compression Efficiency vs Training Amount')
    for i, v in enumerate(bpb_values):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # PPL by epoch
    axes[1].bar(range(len(epochs)), ppl_values, color=colors)
    axes[1].set_xticks(range(len(epochs)))
    axes[1].set_xticklabels([f'{e} epoch{"s" if e > 1 else ""}' for e in epochs])
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Predictive Quality vs Training Amount')
    for i, v in enumerate(ppl_values):
        axes[1].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'compression_training_amount_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {figures_dir / 'compression_training_amount_effect.png'}")
    
    # Print improvements
    if len(bpb_values) >= 2:
        improvement_1_to_5 = (bpb_values[0] - bpb_values[1]) / bpb_values[0] * 100
        print(f"BPB improvement from 1→5 epochs: {improvement_1_to_5:.1f}%")
    if len(bpb_values) >= 3:
        improvement_5_to_22 = (bpb_values[1] - bpb_values[2]) / bpb_values[1] * 100
        print(f"BPB improvement from 5→22 epochs: {improvement_5_to_22:.1f}%")


def print_architecture_comparison(all_metrics: Dict):
    """Compare 1-stage vs 2-stage under same conditions."""
    arch_comparisons = [
        ('PI1M_concat_5epoch', 'PI1M_concat_5epoch_2stage'),
        ('MOSES_concat_5epoch', 'MOSES_concat_5epoch_2stage'),
    ]
    
    print("\n=== Architecture Comparison: 1-Stage vs 2-Stage ===")
    for one_stage, two_stage in arch_comparisons:
        m1 = all_metrics.get(one_stage)
        m2 = all_metrics.get(two_stage)
        
        dataset = MODELS[one_stage]['dataset']
        
        if m1 and m2:
            bpb_improvement = (m1['final_bpb'] - m2['final_bpb']) / m1['final_bpb'] * 100
            ppl_improvement = (m1['final_perplexity'] - m2['final_perplexity']) / m1['final_perplexity'] * 100
            
            print(f"\n{dataset}:")
            print(f"  1-Stage BPB: {m1['final_bpb']:.3f} → 2-Stage BPB: {m2['final_bpb']:.3f} ({bpb_improvement:+.1f}%)")
            print(f"  1-Stage PPL: {m1['final_perplexity']:.2f} → 2-Stage PPL: {m2['final_perplexity']:.2f} ({ppl_improvement:+.1f}%)")
        elif m1 and not m2:
            print(f"\n{dataset}:")
            print(f"  1-Stage BPB: {m1['final_bpb']:.3f}, PPL: {m1['final_perplexity']:.2f}")
            print(f"  2-Stage: ⏳ Training pending")
        else:
            print(f"\n{dataset}: Data not available")


def save_metrics_to_csv(all_metrics: Dict, summary_df: pd.DataFrame):
    """Save metrics to CSV for future reference."""
    # Save summary table
    summary_path = data_dir / 'compression_metrics_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    # Save detailed metrics as JSON
    detailed = {}
    for name, metrics in all_metrics.items():
        if metrics is not None:
            detailed[name] = {
                'final_bpb': metrics['final_bpb'],
                'final_perplexity': metrics['final_perplexity'],
                'best_bpb': metrics['best_bpb'],
                'best_perplexity': metrics['best_perplexity'],
                'total_training_bytes': int(metrics['total_training_bytes']),
                'dataset': metrics['dataset'],
                'architecture': metrics['architecture'],
                'epochs': metrics['epochs'],
                'concatenate': metrics['concatenate'],
            }
    
    json_path = data_dir / 'compression_metrics_detailed.json'
    with open(json_path, 'w') as f:
        json.dump(detailed, f, indent=2)
    print(f"Saved: {json_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("COMPRESSION METRICS ANALYSIS")
    print("Bits-Per-Byte (BPB) & Perplexity (PPL) for H-Net on SMILES")
    print("=" * 60 + "\n")
    
    # Load all models
    all_metrics, all_histories = load_all_models()
    
    # Create summary table
    print("\n=== Compression Metrics Summary ===")
    summary_df = create_summary_table(all_metrics)
    print(summary_df.to_string(index=False))
    
    # Generate plots
    print("\n=== Generating Plots ===")
    plot_bpb_training_dynamics(all_histories)
    plot_bpb_comparison(all_metrics)
    plot_training_amount_effect(all_metrics)
    
    # Architecture comparison
    print_architecture_comparison(all_metrics)
    
    # Save data
    print("\n=== Saving Data ===")
    save_metrics_to_csv(all_metrics, summary_df)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    valid_metrics = {k: v for k, v in all_metrics.items() if v is not None}
    if valid_metrics:
        best_model = min(valid_metrics.items(), key=lambda x: x[1]['final_bpb'])
        print(f"\n🏆 Best Compression: {best_model[0]}")
        print(f"   BPB: {best_model[1]['final_bpb']:.3f}")
        print(f"   PPL: {best_model[1]['final_perplexity']:.2f}")
        
        print("\n📈 Key Observations:")
        print("   - BPB measures how efficiently H-Net compresses SMILES")
        print("   - PPL measures how confidently the model predicts next bytes")
        print("   - Lower values = better performance")
        print(f"   - Theoretical max BPB: 8.0 (random prediction)")
    
    print(f"\n📁 Figures saved to: {figures_dir}")
    print(f"📁 Data saved to: {data_dir}")


if __name__ == '__main__':
    main()

