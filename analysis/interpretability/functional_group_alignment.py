#!/usr/bin/env python3
"""
Functional Group Alignment Analysis for H-Net Tokenization.

This script analyzes how well H-Net tokens align with common chemical
functional groups. For each functional group, we check:
1. Whether H-Net captures it as a single token
2. Whether it's split across multiple tokens
3. Comparison with SmilesPE tokenization

Uses SMARTS patterns for functional group detection.
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import pandas as pd

# Paths
DATA_DIR = Path(__file__).parent / 'data'
HNET_RESULTS_DIR = Path(__file__).parent.parent / 'data' / 'hnet_results'
SMILESPE_RESULTS_DIR = Path(__file__).parent.parent / 'data' / 'smilesPE_results'


# Define functional groups by their SMILES substrings
# Each entry is: (group_name, smiles_patterns, description)
FUNCTIONAL_GROUPS = {
    # Oxygen groups
    'carbonyl': {
        'patterns': ['=O', 'C=O'],
        'description': 'C=O (carbonyl oxygen)',
    },
    'hydroxyl': {
        'patterns': ['O', 'OH'],
        'description': '-OH (hydroxyl)',
    },
    'carboxyl': {
        'patterns': ['C(=O)O', 'COOH', '(=O)O'],
        'description': '-COOH (carboxylic acid)',
    },
    'ester': {
        'patterns': ['C(=O)O', 'COO', 'OC(=O)'],
        'description': '-COO- (ester)',
    },
    'ether': {
        'patterns': ['COC', 'OC'],
        'description': '-C-O-C- (ether)',
    },
    
    # Nitrogen groups
    'amine': {
        'patterns': ['N', 'NH', 'NH2', 'N('],
        'description': '-NH2/-NHR/-NR2 (amine)',
    },
    'amide': {
        'patterns': ['C(=O)N', 'NC=O', 'NC(=O)'],
        'description': '-CONH- (amide)',
    },
    'nitrile': {
        'patterns': ['C#N', '#N'],
        'description': '-C≡N (nitrile)',
    },
    
    # Sulfur groups
    'thiol': {
        'patterns': ['S', 'SH'],
        'description': '-SH (thiol)',
    },
    'sulfone': {
        'patterns': ['S(=O)(=O)', 'SO2'],
        'description': '-SO2- (sulfone)',
    },
    
    # Aromatic patterns
    'benzene': {
        'patterns': ['c1ccccc1', 'c1ccc'],
        'description': 'Benzene ring',
    },
    'phenyl': {
        'patterns': ['c1ccc', 'ccc', 'cc'],
        'description': 'Aromatic carbons',
    },
    
    # Halogens
    'fluorine': {
        'patterns': ['F'],
        'description': '-F (fluorine)',
    },
    'chlorine': {
        'patterns': ['Cl'],
        'description': '-Cl (chlorine)',
    },
    'trifluoromethyl': {
        'patterns': ['C(F)(F)F', 'CF3', 'F)(F)F'],
        'description': '-CF3 (trifluoromethyl)',
    },
    
    # Alkyl chains
    'methyl': {
        'patterns': ['C', 'CH3'],
        'description': '-CH3 (methyl)',
    },
    'ethyl': {
        'patterns': ['CC', 'CCH3'],
        'description': '-C2H5 (ethyl)',
    },
    'propyl': {
        'patterns': ['CCC'],
        'description': '-C3H7 (propyl)',
    },
}


def find_pattern_in_smiles(smiles: str, patterns: List[str]) -> List[Tuple[int, int, str]]:
    """
    Find all occurrences of patterns in a SMILES string.
    
    Returns:
        List of (start_pos, end_pos, matched_pattern) tuples
    """
    occurrences = []
    for pattern in patterns:
        start = 0
        while True:
            pos = smiles.find(pattern, start)
            if pos == -1:
                break
            occurrences.append((pos, pos + len(pattern), pattern))
            start = pos + 1
    
    # Sort by position and remove overlapping matches (keep longest)
    occurrences.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    
    # Remove overlaps
    filtered = []
    last_end = -1
    for start, end, pattern in occurrences:
        if start >= last_end:
            filtered.append((start, end, pattern))
            last_end = end
    
    return filtered


def check_token_alignment(
    smiles: str, 
    tokens: List[str], 
    pattern_start: int, 
    pattern_end: int
) -> Dict:
    """
    Check if a functional group pattern aligns with token boundaries.
    
    Returns:
        Dictionary with alignment analysis
    """
    # Reconstruct token positions
    token_positions = []
    pos = 0
    for token in tokens:
        token_positions.append((pos, pos + len(token), token))
        pos += len(token)
    
    # Find tokens that overlap with the pattern
    overlapping_tokens = []
    for t_start, t_end, token in token_positions:
        if t_start < pattern_end and t_end > pattern_start:
            overlapping_tokens.append({
                'token': token,
                'start': t_start,
                'end': t_end,
            })
    
    # Check alignment quality
    if not overlapping_tokens:
        return {
            'alignment': 'not_found',
            'num_tokens': 0,
            'tokens': [],
        }
    
    if len(overlapping_tokens) == 1:
        t = overlapping_tokens[0]
        if t['start'] <= pattern_start and t['end'] >= pattern_end:
            # Pattern is fully contained in one token
            if t['start'] == pattern_start and t['end'] == pattern_end:
                return {
                    'alignment': 'exact',
                    'num_tokens': 1,
                    'tokens': [t['token']],
                }
            else:
                return {
                    'alignment': 'contained',
                    'num_tokens': 1,
                    'tokens': [t['token']],
                }
    
    # Pattern spans multiple tokens
    return {
        'alignment': 'split',
        'num_tokens': len(overlapping_tokens),
        'tokens': [t['token'] for t in overlapping_tokens],
    }


def analyze_sample(sample: Dict, functional_groups: Dict) -> Dict:
    """
    Analyze functional group alignment for a single sample.
    """
    smiles = sample.get('text', '')
    tokens = sample.get('tokens', [])
    
    results = {}
    
    for group_name, group_info in functional_groups.items():
        patterns = group_info['patterns']
        occurrences = find_pattern_in_smiles(smiles, patterns)
        
        if not occurrences:
            results[group_name] = {
                'found': False,
                'count': 0,
                'alignments': [],
            }
            continue
        
        alignments = []
        for start, end, pattern in occurrences:
            alignment = check_token_alignment(smiles, tokens, start, end)
            alignment['pattern'] = pattern
            alignments.append(alignment)
        
        results[group_name] = {
            'found': True,
            'count': len(occurrences),
            'alignments': alignments,
        }
    
    return results


def aggregate_alignment_stats(all_results: List[Dict]) -> Dict:
    """
    Aggregate alignment statistics across all samples.
    """
    stats = {}
    
    for group_name in FUNCTIONAL_GROUPS.keys():
        total_occurrences = 0
        exact_alignments = 0
        contained_alignments = 0
        split_alignments = 0
        not_found = 0
        
        for result in all_results:
            group_result = result.get(group_name, {})
            if not group_result.get('found', False):
                continue
            
            for alignment in group_result.get('alignments', []):
                total_occurrences += 1
                if alignment['alignment'] == 'exact':
                    exact_alignments += 1
                elif alignment['alignment'] == 'contained':
                    contained_alignments += 1
                elif alignment['alignment'] == 'split':
                    split_alignments += 1
                else:
                    not_found += 1
        
        if total_occurrences > 0:
            stats[group_name] = {
                'total_occurrences': total_occurrences,
                'exact_pct': round(exact_alignments / total_occurrences * 100, 1),
                'contained_pct': round(contained_alignments / total_occurrences * 100, 1),
                'split_pct': round(split_alignments / total_occurrences * 100, 1),
                'single_token_pct': round((exact_alignments + contained_alignments) / total_occurrences * 100, 1),
            }
        else:
            stats[group_name] = {
                'total_occurrences': 0,
                'exact_pct': 0,
                'contained_pct': 0,
                'split_pct': 0,
                'single_token_pct': 0,
            }
    
    return stats


def run_alignment_analysis(pkl_file: Path, max_samples: int = 1000) -> Dict:
    """
    Run functional group alignment analysis on a tokenization pkl file.
    """
    print(f"Loading {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    samples = data[:max_samples] if len(data) > max_samples else data
    print(f"Analyzing {len(samples)} samples...")
    
    all_results = []
    for i, sample in enumerate(samples):
        result = analyze_sample(sample, FUNCTIONAL_GROUPS)
        all_results.append(result)
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")
    
    stats = aggregate_alignment_stats(all_results)
    return stats


def main():
    """Main function to run functional group alignment analysis."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Models to analyze
    models = {
        'PI1M_concat_22epoch': HNET_RESULTS_DIR / 'PI1M_concat_22epoch_tokenization.pkl',
        'PI1M_concat_5epoch': HNET_RESULTS_DIR / 'PI1M_concat_5epoch_tokenization.pkl',
        'MOSES_concat_5epoch': HNET_RESULTS_DIR / 'MOSES_concat_5epoch_tokenization.pkl',
        'SmilesPE_PI1M': SMILESPE_RESULTS_DIR / 'SmilesPE_PI1M_tokenization.pkl',
        'SmilesPE_MOSES': SMILESPE_RESULTS_DIR / 'SmilesPE_MOSES_tokenization.pkl',
    }
    
    all_stats = {}
    
    for model_name, pkl_path in models.items():
        if pkl_path.exists():
            print(f"\n=== Analyzing {model_name} ===")
            stats = run_alignment_analysis(pkl_path, max_samples=1000)
            all_stats[model_name] = stats
        else:
            print(f"Warning: {pkl_path} not found, skipping {model_name}")
    
    # Save raw stats
    output_path = DATA_DIR / 'functional_group_alignment.json'
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n\nSaved results to {output_path}")
    
    # Create comparison table
    comparison_rows = []
    for group_name, group_info in FUNCTIONAL_GROUPS.items():
        row = {
            'functional_group': group_name,
            'description': group_info['description'],
        }
        
        for model_name in all_stats.keys():
            stats = all_stats[model_name].get(group_name, {})
            row[f'{model_name}_occurrences'] = stats.get('total_occurrences', 0)
            row[f'{model_name}_single_token_pct'] = stats.get('single_token_pct', 0)
        
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = DATA_DIR / 'functional_group_alignment.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison table to {comparison_path}")
    
    # Print summary
    print("\n=== Functional Group Alignment Summary ===")
    print("\n% of occurrences captured in single token:")
    print(f"{'Functional Group':<20} ", end='')
    for model_name in all_stats.keys():
        print(f"{model_name:<25} ", end='')
    print()
    print("-" * 120)
    
    for group_name in ['carbonyl', 'hydroxyl', 'carboxyl', 'amine', 'amide', 'benzene', 'ethyl', 'trifluoromethyl']:
        if group_name in FUNCTIONAL_GROUPS:
            print(f"{group_name:<20} ", end='')
            for model_name in all_stats.keys():
                stats = all_stats[model_name].get(group_name, {})
                pct = stats.get('single_token_pct', 0)
                total = stats.get('total_occurrences', 0)
                print(f"{pct:5.1f}% (n={total:<4}) ", end='')
            print()
    
    return all_stats


if __name__ == '__main__':
    stats = main()




