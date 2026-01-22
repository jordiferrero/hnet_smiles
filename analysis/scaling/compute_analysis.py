#!/usr/bin/env python3
"""
Scaling Analysis: Compute FLOPs and Performance vs Training Compute.

This script analyzes how tokenization quality scales with training compute
using existing compression metrics data.

Key outputs:
1. FLOPs calculation for each model
2. Performance vs compute curves
3. Power-law fitting for scaling behavior
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

# Paths
DATA_DIR = Path(__file__).parent / 'data'
ANALYSIS_DATA_DIR = Path(__file__).parent.parent / 'data'
STATS_DIR = ANALYSIS_DATA_DIR / 'statistics'


def load_compression_metrics():
    """Load existing compression metrics."""
    csv_path = ANALYSIS_DATA_DIR / 'compression_metrics_summary.csv'
    df = pd.read_csv(csv_path)
    return df


def load_all_models_comparison():
    """Load all models comparison data."""
    csv_path = ANALYSIS_DATA_DIR / 'all_models_comparison.csv'
    df = pd.read_csv(csv_path)
    return df


def estimate_flops(model_params: int, training_bytes: int, seq_length: int = 256) -> float:
    """
    Estimate training FLOPs for a model.
    
    Approximation based on transformer training:
    FLOPs ≈ 6 * params * tokens_processed
    (2x forward, 4x backward per token)
    
    For byte-level models, tokens ≈ bytes
    
    Args:
        model_params: Number of model parameters
        training_bytes: Total bytes processed during training
        seq_length: Sequence length (for accurate token count)
    
    Returns:
        Estimated FLOPs
    """
    # Each byte is a token in byte-level models
    tokens_processed = training_bytes
    
    # Standard approximation: 6 * params * tokens
    flops = 6 * model_params * tokens_processed
    
    return flops


def compute_scaling_metrics():
    """Compute scaling metrics for all models."""
    # Load data
    compression_df = load_compression_metrics()
    comparison_df = load_all_models_comparison()
    
    print("Loaded compression metrics:")
    print(compression_df.to_string())
    print()
    
    # Model parameters (from metadata)
    MODEL_PARAMS = 350_000_000  # 350M parameters (approximate)
    
    # Focus on PI1M models with different training amounts
    scaling_models = compression_df[
        (compression_df['Dataset'] == 'PI1M') & 
        (compression_df['Concatenation'] == 'Yes') &
        (compression_df['Architecture'] == '1-stage')
    ].copy()
    
    print("Scaling models (PI1M, concat, 1-stage):")
    print(scaling_models.to_string())
    print()
    
    # Calculate FLOPs for each model
    results = []
    for _, row in scaling_models.iterrows():
        # Parse training bytes (remove commas if present)
        training_bytes_str = str(row['Training Bytes']).replace(',', '')
        training_bytes = int(training_bytes_str)
        
        flops = estimate_flops(MODEL_PARAMS, training_bytes)
        
        # Get token stats from comparison data
        model_key = f"PI1M_concat_{row['Epochs']}epoch"
        token_stats = comparison_df[comparison_df['Model'] == model_key]
        
        if len(token_stats) > 0:
            unique_tokens = token_stats['Unique Tokens'].values[0]
            tokens_per_smiles = token_stats['Avg Tokens/SMILES'].values[0]
            mean_token_length = token_stats['Mean Token Length'].values[0]
        else:
            unique_tokens = None
            tokens_per_smiles = None
            mean_token_length = None
        
        results.append({
            'model': row['Model'],
            'epochs': row['Epochs'],
            'training_bytes': training_bytes,
            'flops': flops,
            'flops_log10': np.log10(flops),
            'bpb': row['Best BPB'],
            'ppl': row['Final PPL'],
            'unique_tokens': unique_tokens,
            'tokens_per_smiles': tokens_per_smiles,
            'mean_token_length': mean_token_length,
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate improvement ratios
    if len(results_df) > 0:
        baseline = results_df.iloc[0]
        results_df['bpb_improvement'] = (baseline['bpb'] - results_df['bpb']) / baseline['bpb'] * 100
        results_df['efficiency_improvement'] = (baseline['tokens_per_smiles'] - results_df['tokens_per_smiles']) / baseline['tokens_per_smiles'] * 100
        results_df['vocab_growth'] = (results_df['unique_tokens'] - baseline['unique_tokens']) / baseline['unique_tokens'] * 100
    
    return results_df


def fit_power_law(x, y):
    """Fit a power law y = a * x^b using log-log regression."""
    log_x = np.log(x)
    log_y = np.log(y)
    
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_x, log_y)
    
    a = np.exp(intercept)
    b = slope
    
    return {
        'a': a,
        'b': b,
        'r_squared': r_value ** 2,
        'equation': f'y = {a:.4f} * x^{b:.4f}',
    }


def main():
    """Main function to compute scaling analysis."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== Scaling Analysis ===\n")
    
    # Compute scaling metrics
    results_df = compute_scaling_metrics()
    
    print("\n=== Scaling Results ===")
    print(results_df.to_string())
    
    # Save results
    output_path = DATA_DIR / 'compute_efficiency.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Fit power law for BPB vs FLOPs
    if len(results_df) >= 3:
        flops = results_df['flops'].values
        bpb = results_df['bpb'].values
        
        power_law = fit_power_law(flops, bpb)
        print(f"\nPower law fit for BPB vs FLOPs:")
        print(f"  {power_law['equation']}")
        print(f"  R² = {power_law['r_squared']:.4f}")
        
        # Save power law fit
        with open(DATA_DIR / 'power_law_fit.json', 'w') as f:
            json.dump(power_law, f, indent=2)
    
    # Print summary
    print("\n=== Scaling Summary ===")
    if len(results_df) >= 2:
        first = results_df.iloc[0]
        last = results_df.iloc[-1]
        
        compute_increase = last['flops'] / first['flops']
        bpb_improvement = (first['bpb'] - last['bpb']) / first['bpb'] * 100
        efficiency_improvement = (first['tokens_per_smiles'] - last['tokens_per_smiles']) / first['tokens_per_smiles'] * 100
        vocab_growth = (last['unique_tokens'] - first['unique_tokens']) / first['unique_tokens'] * 100
        
        print(f"Compute increase: {compute_increase:.1f}x ({first['epochs']} -> {last['epochs']} epochs)")
        print(f"BPB improvement: {bpb_improvement:.1f}% ({first['bpb']:.3f} -> {last['bpb']:.3f})")
        print(f"Efficiency improvement: {efficiency_improvement:.1f}% ({first['tokens_per_smiles']:.1f} -> {last['tokens_per_smiles']:.1f} tokens/SMILES)")
        print(f"Vocabulary growth: {vocab_growth:.1f}% ({int(first['unique_tokens'])} -> {int(last['unique_tokens'])} unique tokens)")
    
    return results_df


if __name__ == '__main__':
    results = main()







