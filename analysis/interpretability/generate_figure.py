#!/usr/bin/env python3
"""
Generate Interpretability Analysis Figure for Publication.

Creates a multi-panel figure showing:
(a) Token category distribution
(b) Top tokens by category  
(c) Atom boundary respect rates
(d) Functional group alignment comparison

Output: publication_latex/figures/interpretability_analysis.pdf
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

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

# Color palette
COLORS = {
    'aliphatic': '#2ecc71',      # Green
    'aromatic': '#9b59b6',       # Purple
    'aromatic_ring': '#8e44ad', # Dark purple
    'functional_group': '#e74c3c',  # Red
    'syntax': '#3498db',         # Blue
    'bond': '#f39c12',           # Orange
    'ring_closure': '#1abc9c',   # Teal
    'polymer_syntax': '#e67e22', # Dark orange
    'mixed': '#95a5a6',          # Gray
    'complex': '#7f8c8d',        # Dark gray
    'element': '#16a085',        # Dark teal
}


def load_data():
    """Load all analysis data."""
    # Token annotations
    annotations = pd.read_csv(DATA_DIR / 'token_annotations.csv')
    
    # Atom boundary stats
    with open(DATA_DIR / 'atom_boundary_stats.json', 'r') as f:
        boundary_stats = json.load(f)
    
    # Functional group alignment
    with open(DATA_DIR / 'functional_group_alignment.json', 'r') as f:
        fg_alignment = json.load(f)
    
    return annotations, boundary_stats, fg_alignment


def plot_category_distribution(ax, annotations):
    """Plot (a) Token category distribution as horizontal bar chart."""
    category_counts = annotations['category'].value_counts()
    
    # Merge small categories
    main_categories = ['aliphatic', 'aromatic_ring', 'functional_group', 
                       'syntax', 'polymer_syntax', 'aromatic', 'bond', 'ring_closure']
    
    counts = []
    labels = []
    colors = []
    
    for cat in main_categories:
        if cat in category_counts:
            counts.append(category_counts[cat])
            labels.append(cat.replace('_', ' ').title())
            colors.append(COLORS.get(cat, '#95a5a6'))
    
    # Add "other" for remaining
    other_count = sum(category_counts[c] for c in category_counts.index if c not in main_categories)
    if other_count > 0:
        counts.append(other_count)
        labels.append('Other')
        colors.append('#95a5a6')
    
    y_pos = np.arange(len(labels))
    
    bars = ax.barh(y_pos, counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Number of Tokens')
    ax.set_title('(a) Token Category Distribution')
    ax.invert_yaxis()
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{count}', va='center', fontsize=8)
    
    ax.set_xlim(0, max(counts) * 1.15)


def plot_top_tokens_by_category(ax, annotations):
    """Plot (b) Example tokens by category."""
    categories = ['aliphatic', 'aromatic_ring', 'functional_group', 'syntax', 'polymer_syntax']
    
    text_content = []
    for cat in categories:
        cat_tokens = annotations[annotations['category'] == cat].head(5)
        tokens = [f"'{t}'" for t in cat_tokens['token'].values]
        text_content.append(f"{cat.replace('_', ' ').title()}:\n  {', '.join(tokens[:4])}")
    
    ax.axis('off')
    ax.set_title('(b) Top Tokens by Category')
    
    # Create text box
    text = '\n\n'.join(text_content)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def plot_atom_boundary_rates(ax, boundary_stats):
    """Plot (c) Atom boundary respect rates comparison."""
    models = list(boundary_stats.keys())
    
    # Cleaner label mapping (aligned with paper wording)
    label_map = {
        'PI1M_concat_22epoch': 'Polymer\n1B bytes',
        'PI1M_concat_5epoch': 'Polymer\n340M bytes',
        'MOSES_concat_5epoch': 'Molecular\n340M bytes',
        'PI1M_concat_5epoch_2stage': 'Polymer\n2-stage',
    }
    
    # Prepare data
    model_labels = []
    fully_respects = []
    splits_atom = []
    
    for model in models:
        stats = boundary_stats[model]
        model_labels.append(label_map.get(model, model.replace('_', '\n')))
        fully_respects.append(stats['mean_fully_respects_rate'] * 100)
        splits_atom.append(stats['mean_splits_atom_rate'] * 100)
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fully_respects, width, label='Fully Respects', 
                   color='#27ae60', edgecolor='white')
    bars2 = ax.bar(x + width/2, splits_atom, width, label='Splits Atom',
                   color='#e74c3c', edgecolor='white')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('(c) Atom Boundary Respect')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=7)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=7)


def plot_functional_group_alignment(ax, fg_alignment):
    """Plot (d) Functional group alignment comparison H-Net vs SmilesPE."""
    groups = ['carbonyl', 'hydroxyl', 'carboxyl', 'amine', 'amide', 'benzene', 'ethyl']
    
    hnet_rates = []
    smilespe_rates = []
    
    for group in groups:
        # H-Net (use 22 epoch model)
        hnet_stats = fg_alignment.get('PI1M_concat_22epoch', {}).get(group, {})
        hnet_rates.append(hnet_stats.get('single_token_pct', 0))
        
        # SmilesPE
        spe_stats = fg_alignment.get('SmilesPE_PI1M', {}).get(group, {})
        smilespe_rates.append(spe_stats.get('single_token_pct', 0))
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hnet_rates, width, label='H-Net (22ep)',
                   color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width/2, smilespe_rates, width, label='SmilesPE',
                   color='#e67e22', edgecolor='white')
    
    ax.set_ylabel('Single Token Capture (%)')
    ax.set_title('(d) Functional Group Alignment')
    ax.set_xticks(x)
    ax.set_xticklabels([g.title() for g in groups], rotation=45, ha='right', fontsize=8)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 110)


def create_figure():
    """Create the main interpretability figure."""
    # Load data
    annotations, boundary_stats, fg_alignment = load_data()
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Chemical Interpretability of H-Net Tokens', fontsize=14, y=0.98)
    
    # Plot each panel
    plot_category_distribution(axes[0, 0], annotations)
    plot_top_tokens_by_category(axes[0, 1], annotations)
    plot_atom_boundary_rates(axes[1, 0], boundary_stats)
    plot_functional_group_alignment(axes[1, 1], fg_alignment)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF
    pdf_path = OUTPUT_DIR / 'interpretability_analysis.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved to {pdf_path}")
    
    # Save as PNG
    png_path = OUTPUT_DIR / 'interpretability_analysis.png'
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved to {png_path}")
    
    plt.close()
    
    return fig


def generate_latex_section():
    """Generate LaTeX content for the paper section."""
    # Load summary data
    annotations = pd.read_csv(DATA_DIR / 'token_annotations.csv')
    with open(DATA_DIR / 'atom_boundary_stats.json', 'r') as f:
        boundary_stats = json.load(f)
    
    category_counts = annotations['category'].value_counts()
    total = len(annotations)
    
    # Get key statistics
    best_model = 'PI1M_concat_22epoch'
    fully_respects = boundary_stats[best_model]['mean_fully_respects_rate'] * 100
    splits_atom = boundary_stats[best_model]['mean_splits_atom_rate'] * 100
    
    latex = f"""
\\subsection{{Chemical Interpretability of Learned Tokens}}

To understand what chemical patterns H-Net learns, we analyzed the top 100 most frequent tokens from our best-performing model (PI1M, 22 epochs).

\\textbf{{Token Categories.}} We automatically classified tokens using a hybrid approach combining character pattern heuristics, RDKit validation, and SMARTS functional group matching. Figure~\\ref{{fig:interpretability}}(a,b) shows the distribution: aliphatic patterns ({category_counts.get('aliphatic', 0)}\\%), aromatic rings ({category_counts.get('aromatic_ring', 0)}\\%), functional groups ({category_counts.get('functional_group', 0)}\\%), and syntactic elements ({category_counts.get('syntax', 0)}\\%).

\\textbf{{Atom Boundary Respect.}} We analyzed whether token boundaries align with atom boundaries in the SMILES string. {fully_respects:.1f}\\% of tokens fully respect atom boundaries (both start and end), while only {splits_atom:.1f}\\% split within an atom symbol (e.g., splitting ``Cl'' as ``C'' and ``l''). Figure~\\ref{{fig:interpretability}}(c) compares boundary respect across models.

\\textbf{{Functional Group Alignment.}} We compared H-Net tokens against common functional groups. Simple groups like hydroxyl (-OH) and ethyl (-CC-) are captured as single tokens in $>$99\\% of cases. Complex groups like carboxyl (-COOH) and amide (-CONH-) show lower alignment (25-30\\%), as H-Net often splits these into multiple semantically meaningful pieces (e.g., ``C(=O)'' and ``N'').

\\textbf{{Comparison with SmilesPE.}} Unlike SmilesPE's chemically-derived vocabulary that achieves higher functional group alignment, H-Net discovers patterns bottom-up from data. While SmilesPE captures benzene rings in $>$99\\% of cases as single tokens, H-Net captures only $\\sim$58\\%---but H-Net learns dataset-specific patterns (like polymer attachment markers ``*)'') not present in SmilesPE's fixed vocabulary.
"""
    
    output_path = DATA_DIR / 'interpretability_section.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved LaTeX section to {output_path}")
    
    return latex


def main():
    """Main function."""
    print("Generating interpretability figure...")
    create_figure()
    
    print("\nGenerating LaTeX section...")
    generate_latex_section()
    
    print("\nDone!")


if __name__ == '__main__':
    main()




