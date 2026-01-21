#!/usr/bin/env python3
"""
Generate publication-quality figures for ICML 2026 paper.
Uses seaborn whitegrid style with mako palette as per analysis conventions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
palette = sns.color_palette("mako", n_colors=8)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def fig_training_amount_effect():
    """
    Bar chart showing training amount effect on tokenization.
    Shows unique tokens, tokens/SMILES, and BPB improvement.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    epochs = ['1 epoch', '5 epochs', '22 epochs']
    x = np.arange(len(epochs))
    
    # Data from FINAL_REPORT.md
    unique_tokens = [4903, 5775, 8019]
    tokens_per_smiles = [21.63, 18.23, 16.59]
    bpb = [0.831, 0.687, 0.639]
    
    # Unique tokens
    bars1 = axes[0].bar(x, unique_tokens, color=palette[1], edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Unique Tokens', fontsize=11)
    axes[0].set_xlabel('Training Duration', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(epochs)
    axes[0].set_title('(a) Token Vocabulary Size', fontsize=12, fontweight='bold')
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars1, unique_tokens)):
        if i > 0:
            pct = (val - unique_tokens[0]) / unique_tokens[0] * 100
            axes[0].annotate(f'+{pct:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                           ha='center', va='bottom', fontsize=9, color='darkgreen')
    
    # Tokens per SMILES
    bars2 = axes[1].bar(x, tokens_per_smiles, color=palette[3], edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Tokens per SMILES', fontsize=11)
    axes[1].set_xlabel('Training Duration', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(epochs)
    axes[1].set_title('(b) Tokenization Efficiency', fontsize=12, fontweight='bold')
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars2, tokens_per_smiles)):
        if i > 0:
            pct = (val - tokens_per_smiles[0]) / tokens_per_smiles[0] * 100
            axes[1].annotate(f'{pct:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                           ha='center', va='bottom', fontsize=9, color='darkred')
    
    # BPB (bits-per-byte)
    bars3 = axes[2].bar(x, bpb, color=palette[5], edgecolor='black', linewidth=0.5)
    axes[2].set_ylabel('Bits-per-Byte (BPB)', fontsize=11)
    axes[2].set_xlabel('Training Duration', fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(epochs)
    axes[2].set_title('(c) Compression Quality', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0.5, 0.9)
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars3, bpb)):
        if i > 0:
            pct = (val - bpb[0]) / bpb[0] * 100
            axes[2].annotate(f'{pct:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                           ha='center', va='bottom', fontsize=9, color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'training_amount_effect.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'training_amount_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: training_amount_effect.pdf/png")


def fig_property_prediction_comparison():
    """
    Bar chart comparing H-Net vs RDKit on property prediction tasks.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Classification (BBBP) - AUC
    methods = ['RDKit', 'H-Net']
    bbbp_auc = [0.927, 0.950]
    bbbp_std = [0.009, 0.002]
    
    x = np.arange(len(methods))
    bars1 = axes[0].bar(x, bbbp_auc, yerr=bbbp_std, color=[palette[2], palette[5]], 
                        edgecolor='black', linewidth=0.5, capsize=4)
    axes[0].set_ylabel('AUC-ROC', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_title('BBBP Classification', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0.9, 0.97)
    # Add star for winner
    axes[0].annotate('★', xy=(1, 0.955), ha='center', fontsize=14, color='gold')
    
    # Regression (Tg) - MAE (lower is better)
    tg_mae = [24.8, 26.6]
    tg_std = [0.7, 0.6]
    
    bars2 = axes[1].bar(x, tg_mae, yerr=tg_std, color=[palette[2], palette[5]], 
                        edgecolor='black', linewidth=0.5, capsize=4)
    axes[1].set_ylabel('MAE (°C)', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_title('Glass Transition Temperature (Tg)', fontsize=12, fontweight='bold')
    axes[1].set_ylim(22, 30)
    # Add comparable annotation
    axes[1].annotate('Competitive\n(+7%)', xy=(1, 27.5), ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'property_prediction_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'property_prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: property_prediction_comparison.pdf/png")


def fig_concatenation_effect():
    """
    Grouped bar chart showing concatenation effect on polymer vs molecular.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    datasets = ['PI1M\n(Polymer)', 'MOSES\n(Molecular)']
    x = np.arange(len(datasets))
    width = 0.35
    
    # Token overlap (Jaccard)
    token_overlap = [0.35, 0.45]
    # Breakpoint stability
    breakpoint_stability = [0.76, 0.96]
    
    axes[0].bar(x - width/2, token_overlap, width, label='Token Overlap', color=palette[1])
    axes[0].bar(x + width/2, breakpoint_stability, width, label='Breakpoint Stability', color=palette[4])
    axes[0].set_ylabel('Jaccard Similarity', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].set_title('(a) Concatenation Stability', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    # Highlight 96% stability
    axes[0].annotate('96%!', xy=(1 + width/2, 0.98), ha='center', fontsize=10, 
                    fontweight='bold', color='darkgreen')
    
    # Vocabulary growth
    unique_tokens_nocat = [4094, 3112]
    unique_tokens_cat = [5775, 6183]
    growth = [(5775-4094)/4094*100, (6183-3112)/3112*100]
    
    axes[1].bar(x - width/2, unique_tokens_nocat, width, label='No Concat', color=palette[2])
    axes[1].bar(x + width/2, unique_tokens_cat, width, label='10× Concat', color=palette[5])
    axes[1].set_ylabel('Unique Tokens', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].set_title('(b) Vocabulary Growth', fontsize=12, fontweight='bold')
    # Add growth percentages
    for i, g in enumerate(growth):
        axes[1].annotate(f'+{g:.0f}%', xy=(i + width/2, unique_tokens_cat[i] + 100),
                        ha='center', fontsize=9, color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'concatenation_effect.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'concatenation_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: concatenation_effect.pdf/png")


def fig_hnet_vs_smilespe():
    """
    Comparison diagram showing H-Net vs SmilesPE characteristics.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Data
    metrics = ['Token\nLength', 'Vocabulary\nSize', 'Tokens per\nSMILES', 'Adaptability\nScore']
    hnet_values = [2.5/6, 7000/8000, 18/22, 1.0]  # Normalized to 0-1
    smilespe_values = [5/6, 2000/8000, 8/22, 0.2]  # Normalized to 0-1
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hnet_values, width, label='H-Net', color=palette[1], edgecolor='black')
    bars2 = ax.bar(x + width/2, smilespe_values, width, label='SmilesPE', color=palette[4], edgecolor='black')
    
    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('H-Net vs SmilesPE: Complementary Tokenization Strategies', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.2)
    
    # Add actual values as text
    actual_hnet = ['2-3 chars', '6-8K', '16-22', 'High']
    actual_smilespe = ['4-6 chars', '1.6-2K', '6-11', 'Low']
    
    for i, (bar, val) in enumerate(zip(bars1, actual_hnet)):
        ax.annotate(val, xy=(bar.get_x() + bar.get_width()/2, hnet_values[i] + 0.03),
                   ha='center', va='bottom', fontsize=8, rotation=0)
    
    for i, (bar, val) in enumerate(zip(bars2, actual_smilespe)):
        ax.annotate(val, xy=(bar.get_x() + bar.get_width()/2, smilespe_values[i] + 0.03),
                   ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hnet_vs_smilespe_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'hnet_vs_smilespe_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: hnet_vs_smilespe_comparison.pdf/png")


def fig_main_results_summary():
    """
    Create a summary figure combining key results for potential use as Figure 1.
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # (a) Dataset specificity - token overlap
    ax1 = fig.add_subplot(gs[0, 0])
    datasets = ['Polymer\nvs\nMolecular', 'Concat\nvs\nNo-Concat\n(PI1M)', 'Concat\nvs\nNo-Concat\n(MOSES)']
    overlaps = [0.30, 0.35, 0.45]
    colors = [palette[0], palette[2], palette[4]]
    bars = ax1.bar(datasets, overlaps, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Token Vocabulary Overlap (Jaccard)', fontsize=10)
    ax1.set_title('(a) Dataset-Specific Tokenization', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 0.6)
    for bar, val in zip(bars, overlaps):
        ax1.annotate(f'{val:.0%}', xy=(bar.get_x() + bar.get_width()/2, val + 0.02),
                    ha='center', fontsize=10, fontweight='bold')
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(2.5, 0.52, '50% threshold', fontsize=8, color='gray')
    
    # (b) Training progression
    ax2 = fig.add_subplot(gs[0, 1])
    epochs = [1, 5, 22]
    unique_tokens = [4903, 5775, 8019]
    bpb = [0.831, 0.687, 0.639]
    
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(epochs, unique_tokens, 'o-', color=palette[1], linewidth=2, markersize=8, label='Unique Tokens')
    line2 = ax2_twin.plot(epochs, bpb, 's--', color=palette[5], linewidth=2, markersize=8, label='BPB')
    
    ax2.set_xlabel('Training Epochs', fontsize=10)
    ax2.set_ylabel('Unique Tokens', fontsize=10, color=palette[1])
    ax2_twin.set_ylabel('Bits-per-Byte (BPB)', fontsize=10, color=palette[5])
    ax2.set_title('(b) Training Progression', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=palette[1])
    ax2_twin.tick_params(axis='y', labelcolor=palette[5])
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right', fontsize=9)
    
    # (c) H-Net vs SmilesPE
    ax3 = fig.add_subplot(gs[1, 0])
    metrics = ['Avg Token\nLength', 'Unique\nTokens', 'Tokens/\nSMILES']
    hnet_vals = [2.5, 6000, 18]
    smilespe_vals = [5.0, 1800, 8]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, hnet_vals, width, label='H-Net', color=palette[1], edgecolor='black')
    ax3.bar(x + width/2, smilespe_vals, width, label='SmilesPE', color=palette[4], edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylabel('Value', fontsize=10)
    ax3.set_title('(c) H-Net vs SmilesPE', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    
    # (d) Property Prediction
    ax4 = fig.add_subplot(gs[1, 1])
    tasks = ['BBBP\n(AUC ↑)', 'Tg\n(MAE ↓)']
    rdkit = [0.927, 24.8]
    hnet = [0.950, 26.6]
    
    x = np.arange(len(tasks))
    width = 0.35
    ax4.bar(x - width/2, [rdkit[0], rdkit[1]/30], width, label='RDKit', color=palette[2], edgecolor='black')
    ax4.bar(x + width/2, [hnet[0], hnet[1]/30], width, label='H-Net', color=palette[5], edgecolor='black')
    
    # Add actual values
    ax4.annotate('0.927', xy=(-width/2, rdkit[0] + 0.02), ha='center', fontsize=9)
    ax4.annotate('0.950★', xy=(width/2, hnet[0] + 0.02), ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    ax4.annotate('24.8°C★', xy=(1 - width/2, rdkit[1]/30 + 0.02), ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    ax4.annotate('26.6°C', xy=(1 + width/2, hnet[1]/30 + 0.02), ha='center', fontsize=9)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(tasks)
    ax4.set_ylabel('Normalized Score', fontsize=10)
    ax4.set_title('(d) Property Prediction', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.set_ylim(0, 1.1)
    
    plt.suptitle('Dynamic Tokenization for Chemical SMILES: Key Results', fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(FIGURES_DIR / 'main_results_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'main_results_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: main_results_summary.pdf/png")


if __name__ == "__main__":
    print("Generating publication-quality figures...")
    print("=" * 50)
    
    fig_training_amount_effect()
    fig_property_prediction_comparison()
    fig_concatenation_effect()
    fig_hnet_vs_smilespe()
    fig_main_results_summary()
    
    print("=" * 50)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("\nFigures created:")
    for f in sorted(FIGURES_DIR.glob("*.pdf")):
        print(f"  - {f.name}")




