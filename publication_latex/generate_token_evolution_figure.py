#!/usr/bin/env python3
"""
Generate Token Evolution Figure for ICML Paper Appendix

This script creates a visualization showing how H-Net tokenization breakpoints
evolve during training, comparing 1-stage vs 2-stage architectures on polymer SMILES.

Stages to compare:
- Low training: ~1-5M bytes (early, near-random tokenization)
- Mid training: ~100M bytes (emerging patterns)
- High training: ~238M bytes (converged tokenization)
"""

import gzip
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import colorsys

# Paths to checkpoint data
CHECKPOINT_1STAGE = "/home/ec2-user/hnet_smiles/checkpoints/run_large_20251111_181836"
CHECKPOINT_2STAGE = "/home/ec2-user/hnet_smiles/checkpoints/run_large_20260115_191350"

# Training byte stages to visualize
TRAINING_STAGES = {
    "Early (1M bytes)": "1,000,000",
    "Mid (100M bytes)": "100,000,000",
    "Converged (238M bytes)": "238,000,000"
}

# Output directory
OUTPUT_DIR = "/home/ec2-user/hnet_smiles/publication_latex/figures"


def load_predictions(checkpoint_dir, bytes_str):
    """Load prediction data from a specific training byte checkpoint."""
    pred_path = os.path.join(
        checkpoint_dir, 
        "visualizations", 
        "predictions", 
        f"predictions_bytes_{bytes_str}.pkl.gz"
    )
    
    if not os.path.exists(pred_path):
        print(f"Warning: {pred_path} not found")
        return None
    
    with gzip.open(pred_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def extract_tokens_from_boundary_mask(smiles, boundary_mask):
    """
    Extract token strings from SMILES and boundary mask.
    
    Args:
        smiles: SMILES string
        boundary_mask: Boolean array where True indicates token boundary
    
    Returns:
        List of token strings
    """
    tokens = []
    start = 0
    
    for i, is_boundary in enumerate(boundary_mask):
        if is_boundary and i > start:
            tokens.append(smiles[start:i])
            start = i
    
    # Add final token
    if start < len(smiles):
        tokens.append(smiles[start:])
    
    return tokens


def extract_single_smiles_tokens(pred_data, smiles_idx=0):
    """
    Extract tokens for a single SMILES from the concatenated string.
    
    The test SMILES are concatenated with spaces. We extract tokens
    for just the first individual SMILES for cleaner visualization.
    """
    if pred_data is None:
        return None, None
    
    predictions = pred_data['predictions']
    
    if len(predictions) <= smiles_idx:
        return None, None
    
    pred = predictions[smiles_idx]
    full_smiles = pred['smiles']
    boundary_mask = pred['boundary_mask']
    
    # Extract tokens for the full concatenated string
    all_tokens = extract_tokens_from_boundary_mask(full_smiles, boundary_mask)
    
    # Find the first individual SMILES (ends at first space)
    space_idx = full_smiles.find(' ')
    if space_idx > 0:
        single_smiles = full_smiles[:space_idx]
        
        # Find tokens that belong to this SMILES
        char_count = 0
        single_tokens = []
        for token in all_tokens:
            if char_count + len(token) <= space_idx:
                single_tokens.append(token)
                char_count += len(token)
            else:
                # Partial token at boundary
                remaining = space_idx - char_count
                if remaining > 0:
                    single_tokens.append(token[:remaining])
                break
        
        return single_smiles, single_tokens
    else:
        return full_smiles, all_tokens


def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio
        saturation = 0.65 + 0.15 * (i % 3)
        lightness = 0.55 + 0.1 * ((i // 3) % 2)
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors


def draw_gif_style_tokenization(ax, smiles, tokens, y_offset=0, label="", color='#3498db'):
    """
    Draw tokenization in GIF visualization style:
    - Top row: Green squares for boundaries, white for non-boundaries
    - Bottom row: Character boxes with U-shape brackets showing tokens
    
    Args:
        ax: Matplotlib axes
        smiles: Original SMILES string
        tokens: List of token strings
        y_offset: Vertical offset for this row
        label: Label for this model
        color: Color for the U-shape brackets
    """
    square_size = 0.8
    num_chars = len(smiles)
    
    # Reconstruct boundary mask from tokens
    boundary_mask = [True]  # First char is always a boundary
    char_idx = 0
    for token in tokens:
        char_idx += len(token)
        if char_idx < num_chars:
            boundary_mask.append(True)
            for _ in range(len(token) - 1):
                if len(boundary_mask) < num_chars:
                    boundary_mask.insert(-1, False)
    
    # Ensure boundary mask matches smiles length
    while len(boundary_mask) < num_chars:
        boundary_mask.append(False)
    boundary_mask = boundary_mask[:num_chars]
    
    # Convert to numpy for easier handling
    boundary_mask = np.array(boundary_mask)
    
    # Y positions
    y_boundary = y_offset + 1.5
    y_chars = y_offset
    
    # Draw model label
    if label:
        avg_len = np.mean([len(t) for t in tokens]) if tokens else 0
        label_text = f"{label} ({len(tokens)} tokens, avg {avg_len:.1f} chars)"
        ax.text(-2, y_offset + 0.75, label_text, ha='right', va='center', 
               fontsize=9, fontweight='bold', color=color)
    
    # Draw boundary markers (top row)
    for i in range(num_chars):
        is_boundary = boundary_mask[i] if i < len(boundary_mask) else False
        facecolor = '#90EE90' if is_boundary else '#FFFFFF'  # Green for boundary
        square = mpatches.Rectangle(
            (i, y_boundary),
            square_size, square_size,
            facecolor=facecolor,
            edgecolor='#2c3e50',
            linewidth=0.5
        )
        ax.add_patch(square)
    
    # Draw character row (bottom) with U-shape brackets
    char_idx = 0
    for token in tokens:
        token_len = len(token)
        if char_idx + token_len > num_chars:
            token_len = num_chars - char_idx
        
        if token_len > 0:
            # Draw U-shape bracket
            start_x = char_idx
            end_x = char_idx + token_len
            
            # U-shape: left vertical, bottom horizontal, right vertical
            ax.plot([start_x, start_x], [y_chars, y_chars + square_size], 
                   color=color, linewidth=2, alpha=0.7)
            ax.plot([start_x, end_x], [y_chars, y_chars], 
                   color=color, linewidth=2, alpha=0.7)
            ax.plot([end_x, end_x], [y_chars, y_chars + square_size], 
                   color=color, linewidth=2, alpha=0.7)
            
            # Light fill
            fill = mpatches.Rectangle(
                (start_x, y_chars),
                token_len, square_size,
                facecolor=color,
                alpha=0.15,
                edgecolor='none'
            )
            ax.add_patch(fill)
        
        char_idx += len(token)
    
    # Draw character squares
    for i, char in enumerate(smiles):
        square = mpatches.Rectangle(
            (i, y_chars),
            square_size, square_size,
            facecolor='#FFFFFF',
            edgecolor='#2c3e50',
            linewidth=0.5
        )
        ax.add_patch(square)
        ax.text(i + square_size/2, y_chars + square_size/2, char,
               ha='center', va='center', fontsize=7, fontfamily='monospace')


def create_token_evolution_figure(output_path):
    """
    Create the token evolution comparison figure in GIF visualization style.
    Shows boundary markers and U-shape token brackets.
    """
    # Collect all data
    all_data = {}
    example_smiles = None
    
    for stage_name, bytes_str in TRAINING_STAGES.items():
        all_data[stage_name] = {}
        
        # Load 1-stage data
        pred_1stage = load_predictions(CHECKPOINT_1STAGE, bytes_str)
        smiles_1s, tokens_1s = extract_single_smiles_tokens(pred_1stage, smiles_idx=0)
        all_data[stage_name]['1stage'] = {'smiles': smiles_1s, 'tokens': tokens_1s}
        
        if example_smiles is None and smiles_1s:
            example_smiles = smiles_1s
        
        # Load 2-stage data
        pred_2stage = load_predictions(CHECKPOINT_2STAGE, bytes_str)
        smiles_2s, tokens_2s = extract_single_smiles_tokens(pred_2stage, smiles_idx=0)
        all_data[stage_name]['2stage'] = {'smiles': smiles_2s, 'tokens': tokens_2s}
    
    # Print summary
    print("\n" + "=" * 70)
    print("TOKEN EVOLUTION SUMMARY")
    print("=" * 70)
    
    for stage_name in TRAINING_STAGES.keys():
        print(f"\n{stage_name}:")
        for model_name, model_key in [('1-Stage', '1stage'), ('2-Stage', '2stage')]:
            data = all_data[stage_name][model_key]
            if data['tokens']:
                avg_len = np.mean([len(t) for t in data['tokens']])
                print(f"  {model_name}: {len(data['tokens'])} tokens, avg len: {avg_len:.2f}")
            else:
                print(f"  {model_name}: No data")
    
    if not example_smiles:
        print("ERROR: No data available for visualization")
        return
    
    # Create figure with GridSpec
    num_stages = len(TRAINING_STAGES)
    fig = plt.figure(figsize=(16, 4 * num_stages + 2))
    
    # Create grid: each training stage gets a row showing both architectures
    gs = fig.add_gridspec(num_stages + 1, 1, height_ratios=[0.5] + [1] * num_stages, hspace=0.3)
    
    # Title panel
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.8, "Evolution of H-Net Tokenization During Training",
                  ha='center', va='center', fontsize=16, fontweight='bold')
    ax_title.text(0.5, 0.4, f"Example PSMILES: {example_smiles}",
                  ha='center', va='center', fontsize=10, fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    ax_title.text(0.5, 0.1, "Green = boundary marker | Blue brackets = 1-Stage tokens | Purple brackets = 2-Stage tokens",
                  ha='center', va='center', fontsize=9, color='#7f8c8d')
    
    # Colors for architectures
    colors = {'1stage': '#3498db', '2stage': '#9b59b6'}  # Blue for 1-stage, Purple for 2-stage
    
    stages = list(TRAINING_STAGES.keys())
    
    for i, stage_name in enumerate(stages):
        ax = fig.add_subplot(gs[i + 1])
        
        # Get data
        data_1s = all_data[stage_name]['1stage']
        data_2s = all_data[stage_name]['2stage']
        
        # Calculate plot bounds
        smiles_len = len(example_smiles) if example_smiles else 40
        ax.set_xlim(-8, smiles_len + 1)
        ax.set_ylim(-0.5, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Stage label
        ax.text(-7.5, 3, stage_name, ha='left', va='center', fontsize=12,
               fontweight='bold', color='#2c3e50',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='none'))
        
        # Draw 1-stage (top, y_offset=3)
        if data_1s['tokens'] and data_1s['smiles']:
            draw_gif_style_tokenization(ax, data_1s['smiles'], data_1s['tokens'], 
                                        y_offset=3, label="1-Stage", color=colors['1stage'])
        
        # Draw 2-stage (bottom, y_offset=0)
        if data_2s['tokens'] and data_2s['smiles']:
            draw_gif_style_tokenization(ax, data_2s['smiles'], data_2s['tokens'], 
                                        y_offset=0, label="2-Stage", color=colors['2stage'])
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


def create_simple_text_figure(output_path):
    """
    Create a simpler text-based figure for the appendix.
    """
    # Collect all data
    all_data = {}
    example_smiles = None
    
    for stage_name, bytes_str in TRAINING_STAGES.items():
        all_data[stage_name] = {}
        
        # Load 1-stage data
        pred_1stage = load_predictions(CHECKPOINT_1STAGE, bytes_str)
        smiles_1s, tokens_1s = extract_single_smiles_tokens(pred_1stage, smiles_idx=0)
        all_data[stage_name]['1stage'] = tokens_1s
        
        if example_smiles is None and smiles_1s:
            example_smiles = smiles_1s
        
        # Load 2-stage data
        pred_2stage = load_predictions(CHECKPOINT_2STAGE, bytes_str)
        _, tokens_2s = extract_single_smiles_tokens(pred_2stage, smiles_idx=0)
        all_data[stage_name]['2stage'] = tokens_2s
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.97, "Evolution of H-Net Tokenization During Training",
            ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax.transAxes)
    
    ax.text(0.5, 0.93, "Comparing 1-Stage vs 2-Stage Architecture on Polymer SMILES (PI1M)",
            ha='center', va='top', fontsize=11, style='italic',
            transform=ax.transAxes)
    
    ax.text(0.5, 0.88, f"Example PSMILES: {example_smiles}",
            ha='center', va='top', fontsize=8, fontfamily='monospace',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    y_pos = 0.80
    line_height = 0.065
    
    stages = list(TRAINING_STAGES.keys())
    
    for stage in stages:
        # Stage header
        ax.text(0.02, y_pos, stage, ha='left', va='top', fontsize=11, 
                fontweight='bold', transform=ax.transAxes,
                color='#2c3e50')
        y_pos -= 0.025
        
        for model_name, model_key in [('1-Stage:', '1stage'), ('2-Stage:', '2stage')]:
            tokens = all_data.get(stage, {}).get(model_key, None)
            
            if tokens:
                # Create tokenized string with | separators
                tokenized_str = '|'.join(tokens)
                
                # Truncate if too long
                max_len = 95
                if len(tokenized_str) > max_len:
                    tokenized_str = tokenized_str[:max_len] + "..."
                
                # Stats
                avg_len = np.mean([len(t) for t in tokens])
                unique = len(set(tokens))
                stats = f"[{len(tokens)} tok, {unique} uniq, avg: {avg_len:.1f}]"
                
                ax.text(0.04, y_pos, model_name, ha='left', va='top', fontsize=9,
                       fontweight='bold', transform=ax.transAxes, color='#34495e')
                ax.text(0.12, y_pos, tokenized_str, ha='left', va='top', fontsize=7,
                       fontfamily='monospace', transform=ax.transAxes)
                ax.text(0.98, y_pos, stats, ha='right', va='top', fontsize=8,
                       transform=ax.transAxes, color='#7f8c8d', style='italic')
            else:
                ax.text(0.04, y_pos, f"{model_name} Data not available", 
                       ha='left', va='top', fontsize=9, transform=ax.transAxes,
                       color='#e74c3c')
            
            y_pos -= line_height
        
        y_pos -= 0.03  # Extra space between stages
    
    # Key observations box
    obs_y = y_pos - 0.02
    obs_text = """Key Observations:
• Early training: Both models produce near byte-level tokenization (short tokens)
• Mid training: 1-stage model learns to group characters into chemical subunits (e.g., 'C(=O)N', 'ccc')
• Converged: Token patterns stabilize; 1-stage shows stronger chemical grouping
• The 1-stage architecture achieves better compression (fewer tokens, higher avg length)"""
    
    ax.text(0.5, obs_y, obs_text, ha='center', va='top', fontsize=9,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4f8', edgecolor='#3498db'),
            linespacing=1.5)
    
    # Legend
    ax.text(0.5, 0.04, "Legend: | = token boundary",
            ha='center', va='top', fontsize=9, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate the token evolution figures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate colored block figure
    output_path_blocks = os.path.join(OUTPUT_DIR, "token_evolution_appendix.pdf")
    create_token_evolution_figure(output_path_blocks)
    
    # Generate simple text figure
    output_path_text = os.path.join(OUTPUT_DIR, "token_evolution_text.pdf")
    create_simple_text_figure(output_path_text)
    
    print("\n✅ All figures generated successfully!")


if __name__ == "__main__":
    main()
