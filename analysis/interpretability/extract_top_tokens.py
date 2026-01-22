#!/usr/bin/env python3
"""
Extract top tokens from existing tokenization statistics JSON files.

This script parses the token_frequency data already computed in
analysis/data/statistics/*.json and creates a consolidated CSV
with the top 100 tokens for interpretability analysis.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Paths
STATS_DIR = Path(__file__).parent.parent / 'data' / 'statistics'
OUTPUT_DIR = Path(__file__).parent / 'data'


def load_token_frequencies(stats_file: Path) -> Dict[str, int]:
    """Load token frequencies from a stats JSON file."""
    with open(stats_file, 'r') as f:
        data = json.load(f)
    return data.get('token_frequency', {})


def get_top_tokens(frequencies: Dict[str, int], n: int = 100) -> List[Tuple[str, int]]:
    """Get top N tokens by frequency."""
    sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    return sorted_tokens[:n]


def extract_example_contexts(token: str, tokenization_pkl: Path, max_examples: int = 5) -> List[str]:
    """
    Extract example SMILES contexts where a token appears.
    Uses existing tokenization pkl files.
    """
    import pickle
    
    if not tokenization_pkl.exists():
        return []
    
    with open(tokenization_pkl, 'rb') as f:
        data = pickle.load(f)
    
    examples = []
    for item in data:
        tokens = item.get('tokens', [])
        text = item.get('text', '')
        
        if token in tokens:
            # Find the token position and extract context
            try:
                token_idx = tokens.index(token)
                # Get surrounding context
                start_idx = max(0, token_idx - 2)
                end_idx = min(len(tokens), token_idx + 3)
                context_tokens = tokens[start_idx:end_idx]
                context = '|'.join(context_tokens)
                examples.append(f"...{context}...")
                
                if len(examples) >= max_examples:
                    break
            except ValueError:
                continue
    
    return examples


def main():
    """Main function to extract top tokens from all models."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Models to analyze (focus on best-performing ones)
    models = {
        'PI1M_concat_22epoch': {
            'stats': STATS_DIR / 'PI1M_concat_22epoch_stats.json',
            'pkl': STATS_DIR.parent / 'hnet_results' / 'PI1M_concat_22epoch_tokenization.pkl',
        },
        'PI1M_concat_5epoch': {
            'stats': STATS_DIR / 'PI1M_concat_5epoch_stats.json',
            'pkl': STATS_DIR.parent / 'hnet_results' / 'PI1M_concat_5epoch_tokenization.pkl',
        },
        'MOSES_concat_5epoch': {
            'stats': STATS_DIR / 'MOSES_concat_5epoch_stats.json',
            'pkl': STATS_DIR.parent / 'hnet_results' / 'MOSES_concat_5epoch_tokenization.pkl',
        },
        'PI1M_concat_5epoch_2stage': {
            'stats': STATS_DIR / 'PI1M_concat_5epoch_2stage_stats.json',
            'pkl': STATS_DIR.parent / 'hnet_results' / 'PI1M_concat_5epoch_2stage_tokenization.pkl',
        },
    }
    
    # Extract top 100 tokens for the primary model (22 epoch)
    primary_model = 'PI1M_concat_22epoch'
    primary_stats = models[primary_model]['stats']
    primary_pkl = models[primary_model]['pkl']
    
    print(f"Loading token frequencies from {primary_stats}...")
    frequencies = load_token_frequencies(primary_stats)
    
    print(f"Total unique tokens: {len(frequencies)}")
    
    top_100 = get_top_tokens(frequencies, n=100)
    
    print(f"Extracting example contexts from {primary_pkl}...")
    
    # Build DataFrame
    rows = []
    for rank, (token, freq) in enumerate(top_100, 1):
        # Get example contexts
        examples = extract_example_contexts(token, primary_pkl, max_examples=3)
        
        row = {
            'rank': rank,
            'token': token,
            'frequency': freq,
            'length': len(token),
            'example_1': examples[0] if len(examples) > 0 else '',
            'example_2': examples[1] if len(examples) > 1 else '',
            'example_3': examples[2] if len(examples) > 2 else '',
        }
        rows.append(row)
        
        if rank <= 10:
            print(f"  {rank:3d}. '{token}' (freq={freq}, len={len(token)})")
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = OUTPUT_DIR / 'top_100_tokens.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved top 100 tokens to {output_path}")
    
    # Also create a summary for all models
    print("\n--- Top 10 Tokens per Model ---")
    all_model_data = []
    
    for model_name, paths in models.items():
        if paths['stats'].exists():
            freqs = load_token_frequencies(paths['stats'])
            top_10 = get_top_tokens(freqs, n=10)
            print(f"\n{model_name}:")
            for rank, (token, freq) in enumerate(top_10, 1):
                print(f"  {rank:2d}. '{token}' ({freq})")
                all_model_data.append({
                    'model': model_name,
                    'rank': rank,
                    'token': token,
                    'frequency': freq,
                })
    
    # Save all models comparison
    all_models_df = pd.DataFrame(all_model_data)
    all_models_path = OUTPUT_DIR / 'top_tokens_all_models.csv'
    all_models_df.to_csv(all_models_path, index=False)
    print(f"\nSaved all models comparison to {all_models_path}")
    
    return df


if __name__ == '__main__':
    df = main()







