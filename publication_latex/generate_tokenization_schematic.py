#!/usr/bin/env python3
"""
Generate tokenization schematic figure for ICML 2026 paper.
Shows how H-Net tokenizes a polymer SMILES string step by step.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as path_effects
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Color palette - vibrant chemistry-inspired colors
COLORS = {
    'bytes': '#1a1a2e',         # Dark navy for raw bytes
    'mamba_enc': '#16697a',     # Teal for Mamba encoder
    'transformer': '#82c0cc',   # Light teal for Transformer
    'boundary': '#ff6b6b',      # Coral for boundaries
    'mamba_dec': '#ffa62b',     # Orange for Mamba decoder
    'tokens': '#2ecc71',        # Green for output tokens
    'aromatic': '#9b59b6',      # Purple for aromatic
    'functional': '#e74c3c',    # Red for functional groups
    'attachment': '#3498db',    # Blue for attachment points
    'aliphatic': '#27ae60',     # Green for aliphatic
    'ring': '#f39c12',          # Gold for ring numbers
    'bracket': '#95a5a6',       # Gray for brackets
}


def create_tokenization_schematic():
    """
    Create a schematic showing H-Net tokenizing a polymer SMILES string.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.6, 'H-Net Dynamic Tokenization of Polymer SMILES', 
            fontsize=16, fontweight='bold', ha='center', va='center',
            fontfamily='DejaVu Sans')
    
    # Example PSMILES: [*]CC(=O)OCC[*] (poly(ethylene succinate) repeat unit)
    smiles = "[*]CC(=O)OCC[*]"
    
    # ===== ROW 1: Raw SMILES input =====
    y_row1 = 6.5
    ax.text(0.5, y_row1, 'Input:', fontsize=11, fontweight='bold', ha='left', va='center')
    
    # Draw raw SMILES with color-coded characters
    char_colors = {
        '[': COLORS['bracket'], ']': COLORS['bracket'], '*': COLORS['attachment'],
        'C': COLORS['aliphatic'], 'c': COLORS['aromatic'],
        '(': COLORS['bracket'], ')': COLORS['bracket'],
        '=': COLORS['functional'], 'O': COLORS['functional'],
        '1': COLORS['ring'], '2': COLORS['ring'],
    }
    
    x_start = 2.0
    char_width = 0.35
    for i, char in enumerate(smiles):
        color = char_colors.get(char, COLORS['bytes'])
        box = FancyBboxPatch((x_start + i * char_width, y_row1 - 0.25), 
                             char_width - 0.02, 0.5,
                             boxstyle="round,pad=0.02", 
                             facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x_start + i * char_width + char_width/2 - 0.01, y_row1, char,
               fontsize=14, fontweight='bold', ha='center', va='center', 
               fontfamily='monospace', color=color)
    
    # Legend for character types
    legend_y = 6.5
    legend_x = 9.5
    legend_items = [
        ('Attachment [*]', COLORS['attachment']),
        ('Aliphatic (C)', COLORS['aliphatic']),
        ('Functional (=O)', COLORS['functional']),
        ('Brackets', COLORS['bracket']),
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(Rectangle((legend_x, legend_y - i * 0.4 - 0.1), 0.25, 0.25, 
                               facecolor=color, edgecolor='black', linewidth=0.5))
        ax.text(legend_x + 0.35, legend_y - i * 0.4, label, fontsize=9, va='center')
    
    # ===== ROW 2: Byte-level encoding =====
    y_row2 = 5.3
    ax.text(0.5, y_row2, 'Bytes:', fontsize=11, fontweight='bold', ha='left', va='center')
    
    # Show ASCII values for each character
    for i, char in enumerate(smiles):
        byte_val = ord(char)
        ax.text(x_start + i * char_width + char_width/2 - 0.01, y_row2, str(byte_val),
               fontsize=8, ha='center', va='center', fontfamily='monospace',
               color='#555555')
    
    # Arrow down
    ax.annotate('', xy=(6, 4.7), xytext=(6, 5.0),
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # ===== ROW 3: H-Net Architecture Diagram =====
    y_arch = 4.2
    
    # Mamba Encoder block
    mamba_enc = FancyBboxPatch((1.5, y_arch - 0.4), 2.5, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor=COLORS['mamba_enc'], 
                                edgecolor='black', linewidth=1.5, alpha=0.9)
    ax.add_patch(mamba_enc)
    ax.text(2.75, y_arch, 'Mamba\nEncoder', fontsize=10, fontweight='bold', 
           ha='center', va='center', color='white')
    
    # Arrow
    ax.annotate('', xy=(4.2, y_arch), xytext=(4.0, y_arch),
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # Transformer block with boundary prediction
    transformer = FancyBboxPatch((4.3, y_arch - 0.5), 3.4, 1.0,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLORS['transformer'], 
                                  edgecolor='black', linewidth=1.5, alpha=0.9)
    ax.add_patch(transformer)
    ax.text(6.0, y_arch + 0.15, 'Transformer + Boundary', fontsize=10, fontweight='bold', 
           ha='center', va='center', color='#1a1a2e')
    ax.text(6.0, y_arch - 0.2, 'Prediction', fontsize=10, fontweight='bold',
           ha='center', va='center', color='#1a1a2e')
    
    # Boundary prediction indicator
    boundary_box = FancyBboxPatch((5.2, y_arch - 0.65), 1.6, 0.25,
                                   boxstyle="round,pad=0.02",
                                   facecolor=COLORS['boundary'], 
                                   edgecolor='black', linewidth=1, alpha=0.8)
    ax.add_patch(boundary_box)
    ax.text(6.0, y_arch - 0.53, '⟨boundary scores⟩', fontsize=7, 
           ha='center', va='center', color='white', fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(7.9, y_arch), xytext=(7.7, y_arch),
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # Mamba Decoder block
    mamba_dec = FancyBboxPatch((8.0, y_arch - 0.4), 2.5, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor=COLORS['mamba_dec'], 
                                edgecolor='black', linewidth=1.5, alpha=0.9)
    ax.add_patch(mamba_dec)
    ax.text(9.25, y_arch, 'Mamba\nDecoder', fontsize=10, fontweight='bold', 
           ha='center', va='center', color='white')
    
    # Architecture label
    ax.text(6.0, y_arch + 0.8, 'H-Net Architecture: [m4, [T22], m4]', 
           fontsize=9, ha='center', va='center', style='italic', color='#555')
    
    # Arrow down
    ax.annotate('', xy=(6, 3.2), xytext=(6, 3.5),
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # ===== ROW 4: Boundary Prediction Visualization =====
    y_row4 = 2.7
    ax.text(0.5, y_row4, 'Boundaries:', fontsize=11, fontweight='bold', ha='left', va='center')
    
    # Show boundary predictions (1 = boundary after this char, 0 = no boundary)
    # [*] | CC | (=O) | OCC | [*]
    boundaries = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]  # After each char
    
    for i, (char, is_boundary) in enumerate(zip(smiles, boundaries)):
        # Character
        ax.text(x_start + i * char_width + char_width/2 - 0.01, y_row4 + 0.15, char,
               fontsize=11, ha='center', va='center', fontfamily='monospace',
               color='#333')
        # Boundary indicator
        if i < len(smiles) - 1:
            if is_boundary:
                ax.plot([x_start + (i + 1) * char_width - 0.01, x_start + (i + 1) * char_width - 0.01],
                       [y_row4 - 0.25, y_row4 + 0.35], color=COLORS['boundary'], linewidth=3)
                ax.text(x_start + (i + 1) * char_width - 0.01, y_row4 - 0.35, '|',
                       fontsize=16, ha='center', va='center', color=COLORS['boundary'], fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(6, 1.9), xytext=(6, 2.2),
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # ===== ROW 5: Output Tokens =====
    y_row5 = 1.4
    ax.text(0.5, y_row5, 'Tokens:', fontsize=11, fontweight='bold', ha='left', va='center')
    
    # H-Net learned tokens
    tokens = ['[*]', 'CC', '(=O)', 'OCC', '[*]']
    token_colors = [COLORS['attachment'], COLORS['aliphatic'], COLORS['functional'], 
                    COLORS['aliphatic'], COLORS['attachment']]
    
    token_x = 2.0
    for i, (token, color) in enumerate(zip(tokens, token_colors)):
        token_width = len(token) * 0.35 + 0.2
        box = FancyBboxPatch((token_x, y_row5 - 0.3), token_width, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(token_x + token_width/2, y_row5, token,
               fontsize=12, fontweight='bold', ha='center', va='center',
               fontfamily='monospace', color='white',
               path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        token_x += token_width + 0.15
    
    # Comparison with SmilesPE
    y_compare = 0.5
    ax.text(0.5, y_compare, 'SmilesPE:', fontsize=10, ha='left', va='center', color='#666')
    
    smilespe_tokens = ['[*]', 'CC(=O)O', 'CC', '[*]']
    smilespe_x = 2.0
    for token in smilespe_tokens:
        token_width = len(token) * 0.25 + 0.15
        box = FancyBboxPatch((smilespe_x, y_compare - 0.2), token_width, 0.4,
                             boxstyle="round,pad=0.03",
                             facecolor='#bdc3c7', edgecolor='#7f8c8d', linewidth=1)
        ax.add_patch(box)
        ax.text(smilespe_x + token_width/2, y_compare, token,
               fontsize=9, ha='center', va='center', fontfamily='monospace', color='#333')
        smilespe_x += token_width + 0.1
    
    # Key insight box
    insight_box = FancyBboxPatch((8.5, 0.2), 5.0, 1.4,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#ffeaa7', edgecolor='#fdcb6e', 
                                  linewidth=2, alpha=0.9)
    ax.add_patch(insight_box)
    ax.text(11.0, 1.25, 'Key Insight', fontsize=10, fontweight='bold', ha='center', color='#333')
    ax.text(11.0, 0.85, 'H-Net learns byte-level boundaries', fontsize=9, ha='center', color='#333')
    ax.text(11.0, 0.55, 'that capture chemical substructures', fontsize=9, ha='center', color='#333')
    ax.text(11.0, 0.25, '(functional groups, attachment points)', fontsize=8, ha='center', 
           color='#555', style='italic')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'tokenization_schematic.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'tokenization_schematic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: tokenization_schematic.pdf/png")


if __name__ == "__main__":
    print("Generating tokenization schematic figure...")
    create_tokenization_schematic()
    print(f"Figure saved to: {FIGURES_DIR}")







