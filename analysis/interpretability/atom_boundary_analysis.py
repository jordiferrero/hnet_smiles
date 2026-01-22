#!/usr/bin/env python3
"""
Atom Boundary Analysis for H-Net Tokenization.

This script analyzes whether H-Net token boundaries align with atom boundaries
in the SMILES strings. Uses the existing tokenization pkl files.

Key metrics:
1. Atom Boundary Respect Rate: % of tokens that start/end at atom boundaries
2. Mean Atoms per Token: How many atoms does each token span?
3. Atom-Splitting Rate: % of tokens that split within an atom symbol

Note: Handles RDKit import failures gracefully using heuristic methods.
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import pandas as pd

# Try to import RDKit, fall back to heuristic methods if not available
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except (ImportError, AttributeError):
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available, using heuristic atom boundary detection")

# Paths
DATA_DIR = Path(__file__).parent / 'data'
HNET_RESULTS_DIR = Path(__file__).parent.parent / 'data' / 'hnet_results'

# Two-letter elements that should not be split
TWO_LETTER_ELEMENTS = {
    'Cl', 'Br', 'Si', 'Se', 'As', 'Te', 'Li', 'Na', 'Mg', 'Al', 
    'Ca', 'Fe', 'Cu', 'Zn', 'Sn', 'Pb', 'Bi', 'Ag', 'Au', 'Pt',
    'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'
}

# Single-letter elements
SINGLE_LETTER_ELEMENTS = set('BCNOPSFIHKVYbcnops')

# Pattern for identifying atoms in SMILES
# Matches: [bracket atoms], Cl, Br, two-letter elements, single letters
ATOM_PATTERN = re.compile(
    r'(\[[^\]]+\]|Cl|Br|Si|Se|As|Te|Li|Na|Mg|Al|Ca|Fe|Cu|Zn|Sn|Pb|Bi|Ag|Au|Pt|[BCNOPSFIHKVYbcnops])'
)


def get_atom_positions(smiles: str) -> List[Tuple[int, int, str]]:
    """
    Get the start and end positions of each atom in a SMILES string.
    
    Returns:
        List of (start_pos, end_pos, atom_symbol) tuples
    """
    positions = []
    for match in ATOM_PATTERN.finditer(smiles):
        start = match.start()
        end = match.end()
        symbol = match.group(0)
        positions.append((start, end, symbol))
    return positions


def get_atom_boundaries(smiles: str) -> set:
    """
    Get all character positions that are atom boundaries.
    An atom boundary is where one atom ends and another begins.
    """
    positions = get_atom_positions(smiles)
    boundaries = {0}  # Start is always a boundary
    
    for start, end, _ in positions:
        boundaries.add(start)
        boundaries.add(end)
    
    return boundaries


def analyze_token_boundary(token: str, position_in_smiles: int, smiles: str) -> Dict:
    """
    Analyze whether a token respects atom boundaries.
    
    Args:
        token: The token string
        position_in_smiles: Starting position of token in SMILES
        smiles: The full SMILES string
    
    Returns:
        Dictionary with analysis results
    """
    result = {
        'token': token,
        'length': len(token),
        'start_pos': position_in_smiles,
        'end_pos': position_in_smiles + len(token),
        'respects_start_boundary': False,
        'respects_end_boundary': False,
        'splits_atom': False,
        'atoms_spanned': 0,
        'atom_symbols': [],
    }
    
    # Get atom boundaries for the SMILES
    atom_boundaries = get_atom_boundaries(smiles)
    atom_positions = get_atom_positions(smiles)
    
    # Check if start is at atom boundary
    result['respects_start_boundary'] = position_in_smiles in atom_boundaries
    
    # Check if end is at atom boundary
    end_pos = position_in_smiles + len(token)
    result['respects_end_boundary'] = end_pos in atom_boundaries
    
    # Count atoms spanned by this token
    atoms_in_token = []
    for start, end, symbol in atom_positions:
        # Check if this atom overlaps with the token
        token_start = position_in_smiles
        token_end = position_in_smiles + len(token)
        
        if start < token_end and end > token_start:
            atoms_in_token.append(symbol)
            
            # Check if token splits this atom
            if start < token_start and end > token_start:
                result['splits_atom'] = True
            if start < token_end and end > token_end:
                result['splits_atom'] = True
    
    result['atoms_spanned'] = len(atoms_in_token)
    result['atom_symbols'] = atoms_in_token
    
    # Check for two-letter element splitting
    for elem in TWO_LETTER_ELEMENTS:
        # Token ends with first letter but not the full element
        if token.endswith(elem[0]) and not token.endswith(elem):
            # Check if next char in SMILES would complete the element
            if end_pos < len(smiles) and smiles[end_pos:end_pos+1].lower() == elem[1].lower():
                result['splits_atom'] = True
                break
    
    return result


def analyze_tokenization_sample(sample: Dict) -> Dict:
    """
    Analyze a single tokenization sample for atom boundary respect.
    """
    smiles = sample.get('text', '')
    tokens = sample.get('tokens', [])
    breakpoints = sample.get('breakpoints', [])
    
    if not smiles or not tokens:
        return None
    
    results = {
        'smiles': smiles,
        'num_tokens': len(tokens),
        'token_analyses': [],
        'total_atoms_in_smiles': len(get_atom_positions(smiles)),
    }
    
    # Reconstruct token positions from breakpoints
    position = 0
    for i, token in enumerate(tokens):
        analysis = analyze_token_boundary(token, position, smiles)
        results['token_analyses'].append(analysis)
        position += len(token)
    
    # Aggregate metrics
    analyses = results['token_analyses']
    if analyses:
        results['respects_start_rate'] = sum(1 for a in analyses if a['respects_start_boundary']) / len(analyses)
        results['respects_end_rate'] = sum(1 for a in analyses if a['respects_end_boundary']) / len(analyses)
        results['splits_atom_rate'] = sum(1 for a in analyses if a['splits_atom']) / len(analyses)
        results['mean_atoms_per_token'] = sum(a['atoms_spanned'] for a in analyses) / len(analyses)
        
        # Fully respects boundaries (both start and end)
        results['fully_respects_rate'] = sum(
            1 for a in analyses 
            if a['respects_start_boundary'] and a['respects_end_boundary']
        ) / len(analyses)
    
    return results


def run_boundary_analysis(pkl_file: Path, max_samples: int = 1000) -> Dict:
    """
    Run atom boundary analysis on a tokenization pkl file.
    """
    print(f"Loading {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    samples = data[:max_samples] if len(data) > max_samples else data
    print(f"Analyzing {len(samples)} samples...")
    
    all_results = []
    for i, sample in enumerate(samples):
        result = analyze_tokenization_sample(sample)
        if result:
            all_results.append(result)
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")
    
    # Aggregate statistics
    stats = {
        'total_samples': len(all_results),
        'total_tokens': sum(r['num_tokens'] for r in all_results),
        'mean_tokens_per_smiles': sum(r['num_tokens'] for r in all_results) / len(all_results),
        'mean_atoms_per_smiles': sum(r['total_atoms_in_smiles'] for r in all_results) / len(all_results),
        'mean_respects_start_rate': sum(r['respects_start_rate'] for r in all_results) / len(all_results),
        'mean_respects_end_rate': sum(r['respects_end_rate'] for r in all_results) / len(all_results),
        'mean_fully_respects_rate': sum(r['fully_respects_rate'] for r in all_results) / len(all_results),
        'mean_splits_atom_rate': sum(r['splits_atom_rate'] for r in all_results) / len(all_results),
        'mean_atoms_per_token': sum(r['mean_atoms_per_token'] for r in all_results) / len(all_results),
    }
    
    return stats


def main():
    """Main function to run atom boundary analysis on all models."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Models to analyze
    models = {
        'PI1M_concat_22epoch': HNET_RESULTS_DIR / 'PI1M_concat_22epoch_tokenization.pkl',
        'PI1M_concat_5epoch': HNET_RESULTS_DIR / 'PI1M_concat_5epoch_tokenization.pkl',
        'MOSES_concat_5epoch': HNET_RESULTS_DIR / 'MOSES_concat_5epoch_tokenization.pkl',
        'PI1M_concat_5epoch_2stage': HNET_RESULTS_DIR / 'PI1M_concat_5epoch_2stage_tokenization.pkl',
    }
    
    all_stats = {}
    
    for model_name, pkl_path in models.items():
        if pkl_path.exists():
            print(f"\n=== Analyzing {model_name} ===")
            stats = run_boundary_analysis(pkl_path, max_samples=1000)
            all_stats[model_name] = stats
            
            print(f"\nResults for {model_name}:")
            print(f"  Samples analyzed: {stats['total_samples']}")
            print(f"  Total tokens: {stats['total_tokens']}")
            print(f"  Mean tokens/SMILES: {stats['mean_tokens_per_smiles']:.2f}")
            print(f"  Mean atoms/SMILES: {stats['mean_atoms_per_smiles']:.2f}")
            print(f"  Respects start boundary: {stats['mean_respects_start_rate']*100:.1f}%")
            print(f"  Respects end boundary: {stats['mean_respects_end_rate']*100:.1f}%")
            print(f"  Fully respects boundaries: {stats['mean_fully_respects_rate']*100:.1f}%")
            print(f"  Splits atom: {stats['mean_splits_atom_rate']*100:.1f}%")
            print(f"  Mean atoms/token: {stats['mean_atoms_per_token']:.2f}")
        else:
            print(f"Warning: {pkl_path} not found, skipping {model_name}")
    
    # Save results
    output_path = DATA_DIR / 'atom_boundary_stats.json'
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n\nSaved results to {output_path}")
    
    # Create summary DataFrame
    summary_rows = []
    for model_name, stats in all_stats.items():
        row = {
            'model': model_name,
            'samples': stats['total_samples'],
            'total_tokens': stats['total_tokens'],
            'tokens_per_smiles': round(stats['mean_tokens_per_smiles'], 2),
            'atoms_per_smiles': round(stats['mean_atoms_per_smiles'], 2),
            'respects_start_pct': round(stats['mean_respects_start_rate'] * 100, 1),
            'respects_end_pct': round(stats['mean_respects_end_rate'] * 100, 1),
            'fully_respects_pct': round(stats['mean_fully_respects_rate'] * 100, 1),
            'splits_atom_pct': round(stats['mean_splits_atom_rate'] * 100, 1),
            'atoms_per_token': round(stats['mean_atoms_per_token'], 2),
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = DATA_DIR / 'atom_boundary_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    return all_stats


if __name__ == '__main__':
    stats = main()







