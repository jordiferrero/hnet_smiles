#!/usr/bin/env python3
"""
Generate Scaling Analysis Figure for Publication.

Creates a figure showing:
(a) BPB vs Training Compute (log-log with power law fit)
(b) Token Efficiency Metrics vs Compute

Output: publication_latex/figures/scaling_analysis.pdf
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'publication_latex' / 'figures'

# Style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_data():
    """Load scaling analysis data."""
    df = pd.read_csv(DATA_DIR / 'compute_efficiency.csv')
    
    # Load power law fit if available
    power_law_path = DATA_DIR / 'power_law_fit.json'
    if power_law_path.exists():
        with open(power_law_path, 'r') as f:
            power_law = json.load(f)
    else:
        power_law = None
    
    return df, power_law


def plot_bpb_vs_flops(ax, df, power_law):
    """Plot BPB vs Training FLOPs (log-log scale)."""
    flops = df['flops'].values
    bpb = df['bpb'].values
    epochs = df['epochs'].values
    
    # Plot data points
    ax.scatter(flops, bpb, s=100, c='#3498db', edgecolors='white', 
               linewidth=2, zorder=5, label='H-Net Models')
    
    # Add epoch labels
    for i, (x, y, ep) in enumerate(zip(flops, bpb, epochs)):
        ax.annotate(f'{ep}ep', (x, y), textcoords="offset points",
                    xytext=(10, 5), fontsize=9, color='#2c3e50')
    
    # Plot power law fit if available
    if power_law:
        x_fit = np.logspace(np.log10(flops.min() * 0.5), 
                            np.log10(flops.max() * 2), 100)
        y_fit = power_law['a'] * x_fit ** power_law['b']
        ax.plot(x_fit, y_fit, '--', color='#e74c3c', linewidth=2, alpha=0.7,
                label=f"Power law: BPB ∝ FLOPs$^{{{power_law['b']:.2f}}}$")
    
    ax.set_xscale('log')
    ax.set_xlabel('Training FLOPs')
    ax.set_ylabel('Bits per Byte (BPB)')
    ax.set_title('(a) Compression vs Training Compute')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_ylim(0.55, 0.9)


def plot_efficiency_metrics(ax, df):
    """Plot token efficiency metrics vs compute."""
    flops = df['flops'].values
    tokens_per_smiles = df['tokens_per_smiles'].values
    unique_tokens = df['unique_tokens'].values / 1000  # Convert to thousands
    mean_length = df['mean_token_length'].values
    epochs = df['epochs'].values
    
    # Create twin axis for unique tokens
    ax2 = ax.twinx()
    
    # Plot tokens per SMILES (left axis)
    line1, = ax.plot(flops, tokens_per_smiles, 'o-', color='#27ae60', 
                     linewidth=2, markersize=8, label='Tokens/SMILES')
    
    # Plot unique tokens (right axis)
    line2, = ax2.plot(flops, unique_tokens, 's-', color='#9b59b6',
                      linewidth=2, markersize=8, label='Unique Tokens (K)')
    
    # Add epoch labels
    for i, (x, y1, y2, ep) in enumerate(zip(flops, tokens_per_smiles, unique_tokens, epochs)):
        ax.annotate(f'{ep}ep', (x, y1), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, color='#27ae60', ha='center')
    
    ax.set_xscale('log')
    ax.set_xlabel('Training FLOPs')
    ax.set_ylabel('Tokens per SMILES', color='#27ae60')
    ax2.set_ylabel('Unique Tokens (thousands)', color='#9b59b6')
    ax.set_title('(b) Tokenization Efficiency vs Compute')
    
    # Color the tick labels
    ax.tick_params(axis='y', labelcolor='#27ae60')
    ax2.tick_params(axis='y', labelcolor='#9b59b6')
    
    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    ax.grid(True, alpha=0.3)


def plot_improvement_bars(ax, df):
    """Plot improvement percentages as bar chart."""
    labels = [f"{int(ep)} epochs" for ep in df['epochs'].values]
    
    metrics = {
        'BPB Improvement': df['bpb_improvement'].values,
        'Efficiency Gain': df['efficiency_improvement'].values,
        'Vocab Growth': df['vocab_growth'].values,
    }
    
    x = np.arange(len(labels))
    width = 0.25
    
    colors = ['#3498db', '#27ae60', '#9b59b6']
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[i])
    
    ax.set_xlabel('Training Duration')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('(c) Improvements Relative to 1-Epoch Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Set y-axis limits
    max_val = max(df['vocab_growth'].max(), df['bpb_improvement'].max(), df['efficiency_improvement'].max())
    ax.set_ylim(-5, max_val * 1.15)


def create_figure():
    """Create the main scaling figure."""
    # Load data
    df, power_law = load_data()
    
    # Create figure with 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Scaling Behavior of H-Net Tokenization', fontsize=14, y=1.02)
    
    # Plot each panel
    plot_bpb_vs_flops(axes[0], df, power_law)
    plot_efficiency_metrics(axes[1], df)
    plot_improvement_bars(axes[2], df)
    
    plt.tight_layout()
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF
    pdf_path = OUTPUT_DIR / 'scaling_analysis.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved to {pdf_path}")
    
    # Save as PNG
    png_path = OUTPUT_DIR / 'scaling_analysis.png'
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved to {png_path}")
    
    plt.close()
    
    return fig


def generate_latex_section():
    """Generate LaTeX content for the scaling section."""
    df, power_law = load_data()
    
    # Get key statistics
    first = df.iloc[0]
    last = df.iloc[-1]
    
    compute_increase = last['flops'] / first['flops']
    bpb_improvement = (first['bpb'] - last['bpb']) / first['bpb'] * 100
    efficiency_improvement = (first['tokens_per_smiles'] - last['tokens_per_smiles']) / first['tokens_per_smiles'] * 100
    vocab_growth = (last['unique_tokens'] - first['unique_tokens']) / first['unique_tokens'] * 100
    
    power_exponent = power_law['b'] if power_law else -0.09
    
    latex = f"""
\\subsection{{Scaling Behavior}}

We analyze how tokenization quality scales with training compute (Figure~\\ref{{fig:scaling}}). Using training configurations at 1, 5, and 22 epochs (corresponding to 48M, 238M, and 1048M training bytes), we observe:

\\textbf{{Power-law compression improvement.}} BPB improves from {first['bpb']:.2f} to {last['bpb']:.2f} as training compute increases {compute_increase:.0f}$\\times$, following an approximate power-law relationship: BPB $\\propto$ FLOPs$^{{{power_exponent:.2f}}}$ (R$^2$ = {power_law['r_squared']:.2f}).

\\textbf{{Vocabulary growth.}} Unique tokens increase {vocab_growth:.0f}\\% ({int(first['unique_tokens']):,} $\\rightarrow$ {int(last['unique_tokens']):,}) with extended training, suggesting the model continues to discover specialized patterns rather than overfitting.

\\textbf{{Efficiency saturation.}} Tokens-per-SMILES improves {efficiency_improvement:.0f}\\% ({first['tokens_per_smiles']:.1f} $\\rightarrow$ {last['tokens_per_smiles']:.1f}) with diminishing returns: most improvement occurs in early training.

While we focus on a single model size (350M parameters), these scaling trends suggest that both larger models and longer training could further improve tokenization quality, consistent with scaling laws observed in natural language~\\citep{{kaplan2020scaling}}.
"""
    
    output_path = DATA_DIR / 'scaling_section.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved LaTeX section to {output_path}")
    
    return latex


def main():
    """Main function."""
    print("Generating scaling figure...")
    create_figure()
    
    print("\nGenerating LaTeX section...")
    generate_latex_section()
    
    print("\nDone!")


if __name__ == '__main__':
    main()







