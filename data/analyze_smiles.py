#!/usr/bin/env python3
"""
Analyze SMILES dataset to understand length distribution and determine
concatenation strategy for training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import hnet tokenizer
sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet.utils.tokenizers import ByteTokenizer


def analyze_smiles_dataset(csv_path, sample_size=None):
    """
    Analyze SMILES dataset and compute statistics.
    
    Args:
        csv_path: Path to CSV file with SMILES data
        sample_size: If provided, only analyze first N entries (for quick testing)
    
    Returns:
        dict with statistics
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if sample_size:
        df = df.head(sample_size)
        print(f"Analyzing sample of {len(df)} entries...")
    else:
        print(f"Analyzing full dataset: {len(df)} entries...")
    
    # Extract SMILES strings (first column)
    smiles_col = df.columns[0]
    smiles = df[smiles_col].astype(str)
    
    # Compute length statistics
    lengths = smiles.str.len()
    
    stats = {
        'total_entries': len(smiles),
        'length_stats': {
            'mean': float(lengths.mean()),
            'median': float(lengths.median()),
            'std': float(lengths.std()),
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'q25': float(lengths.quantile(0.25)),
            'q75': float(lengths.quantile(0.75)),
            'q90': float(lengths.quantile(0.90)),
            'q95': float(lengths.quantile(0.95)),
            'q99': float(lengths.quantile(0.99)),
        },
        'length_distribution': lengths.value_counts().sort_index().to_dict(),
    }
    
    # Analyze tokenized lengths (byte-level)
    print("\nAnalyzing byte-level tokenization...")
    tokenizer = ByteTokenizer()
    tokenized_lengths = []
    sample_smiles = smiles.head(1000)  # Sample for tokenization analysis
    
    for smi in sample_smiles:
        encoded = tokenizer.encode([smi], add_bos=False, add_eos=False)[0]
        tokenized_lengths.append(len(encoded['input_ids']))
    
    stats['tokenized_stats'] = {
        'mean': float(np.mean(tokenized_lengths)),
        'median': float(np.median(tokenized_lengths)),
        'std': float(np.std(tokenized_lengths)),
        'min': int(np.min(tokenized_lengths)),
        'max': int(np.max(tokenized_lengths)),
    }
    
    # Determine concatenation strategy
    print("\nAnalyzing concatenation strategies...")
    avg_length = stats['length_stats']['mean']
    median_length = stats['length_stats']['median']
    max_length = stats['length_stats']['max']
    
    # Strategy recommendations
    strategies = []
    
    # Strategy 1: Use individual SMILES
    if median_length < 50:
        strategies.append({
            'name': 'Individual SMILES',
            'description': 'Use each SMILES string individually',
            'pros': ['Simple', 'Natural boundaries', 'No artificial concatenation'],
            'cons': ['Very short sequences', 'May not test chunking well'],
            'recommended': median_length > 20,
        })
    
    # Strategy 2: Concatenate 2-3 SMILES
    target_length = 100
    num_to_concat = int(target_length / avg_length)
    if num_to_concat > 1:
        strategies.append({
            'name': f'Concatenate {num_to_concat} SMILES',
            'description': f'Concatenate {num_to_concat} SMILES with separator',
            'pros': ['Longer sequences', 'Better for chunking analysis'],
            'cons': ['Artificial boundaries', 'More complex preprocessing'],
            'recommended': True,
        })
    
    # Strategy 3: Concatenate 5-10 SMILES
    target_length_long = 200
    num_to_concat_long = int(target_length_long / avg_length)
    if num_to_concat_long > num_to_concat:
        strategies.append({
            'name': f'Concatenate {num_to_concat_long} SMILES',
            'description': f'Concatenate {num_to_concat_long} SMILES for longer sequences',
            'pros': ['Much longer sequences', 'Tests hierarchical chunking'],
            'cons': ['Very artificial', 'May be too long'],
            'recommended': False,
        })
    
    stats['concatenation_strategies'] = strategies
    
    return stats, smiles


def plot_length_distribution(lengths, output_path=None):
    """Plot SMILES length distribution."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram
    axes[0].hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(lengths.mean(), color='r', linestyle='--', label=f'Mean: {lengths.mean():.1f}')
    axes[0].axvline(lengths.median(), color='g', linestyle='--', label=f'Median: {lengths.median():.1f}')
    axes[0].set_xlabel('SMILES Length (characters)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('SMILES Length Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot([lengths], vert=True, patch_artist=True)
    axes[1].set_ylabel('SMILES Length (characters)')
    axes[1].set_title('SMILES Length Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved plot to {output_path}")
    else:
        plt.show()


def print_statistics(stats):
    """Print formatted statistics."""
    print("\n" + "="*60)
    print("SMILES Dataset Statistics")
    print("="*60)
    
    print(f"\nTotal entries: {stats['total_entries']:,}")
    
    print("\nLength Statistics (characters):")
    ls = stats['length_stats']
    print(f"  Mean:   {ls['mean']:.2f}")
    print(f"  Median: {ls['median']:.2f}")
    print(f"  Std:    {ls['std']:.2f}")
    print(f"  Min:    {ls['min']}")
    print(f"  Max:    {ls['max']}")
    print(f"  Q25:    {ls['q25']:.2f}")
    print(f"  Q75:    {ls['q75']:.2f}")
    print(f"  Q90:    {ls['q90']:.2f}")
    print(f"  Q95:    {ls['q95']:.2f}")
    print(f"  Q99:    {ls['q99']:.2f}")
    
    print("\nTokenized Length Statistics (bytes):")
    ts = stats['tokenized_stats']
    print(f"  Mean:   {ts['mean']:.2f}")
    print(f"  Median: {ts['median']:.2f}")
    print(f"  Std:    {ts['std']:.2f}")
    print(f"  Min:    {ts['min']}")
    print(f"  Max:    {ts['max']}")
    
    print("\nConcatenation Strategy Recommendations:")
    for i, strategy in enumerate(stats['concatenation_strategies'], 1):
        rec = "✓ RECOMMENDED" if strategy['recommended'] else ""
        print(f"\n{i}. {strategy['name']} {rec}")
        print(f"   {strategy['description']}")
        print(f"   Pros: {', '.join(strategy['pros'])}")
        print(f"   Cons: {', '.join(strategy['cons'])}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze SMILES dataset')
    parser.add_argument(
        '--csv-path',
        type=str,
        default='../datasets/PI1M/PI1M_v2.csv',
        help='Path to SMILES CSV file'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Analyze only first N entries (for quick testing)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate and save length distribution plot'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save plots and statistics'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    # Analyze dataset
    stats, smiles = analyze_smiles_dataset(csv_path, args.sample_size)
    
    # Print statistics
    print_statistics(stats)
    
    # Generate plot if requested
    if args.plot:
        output_path = Path(args.output_dir) / 'smiles_length_distribution.png'
        lengths = smiles.str.len()
        plot_length_distribution(lengths, output_path)
    
    # Save statistics to JSON
    import json
    stats_output = Path(args.output_dir) / 'smiles_statistics.json'
    # Convert numpy types to native Python types for JSON serialization
    stats_json = json.dumps(stats, indent=2, default=str)
    with open(stats_output, 'w') as f:
        f.write(stats_json)
    print(f"\nSaved statistics to {stats_output}")


if __name__ == '__main__':
    main()

