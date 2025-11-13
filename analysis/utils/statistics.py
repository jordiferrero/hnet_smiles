"""
Utilities for computing and analyzing tokenization statistics.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path


def convert_to_python_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


class TokenStatistics:
    """
    Class to compute and store tokenization statistics.
    """
    
    def __init__(self, tokenization_results: List[Dict]):
        """
        Initialize with tokenization results.
        
        Args:
            tokenization_results: List of tokenization dictionaries
        """
        self.results = tokenization_results
        self.stats = {}
        self._compute_all_statistics()
    
    def _compute_all_statistics(self):
        """Compute all statistics from tokenization results."""
        # Extract all tokens and breakpoints
        all_tokens = []
        all_breakpoint_chars = []
        token_lengths = []
        
        for result in self.results:
            if 'error' in result:
                continue
            
            tokens = result.get('tokens', [])
            all_tokens.extend(tokens)
            token_lengths.extend([len(t) for t in tokens])
            
            bp_chars = result.get('breakpoint_chars', [])
            all_breakpoint_chars.extend(bp_chars)
        
        # Token frequency
        token_counter = Counter(all_tokens)
        
        # Breakpoint character frequency
        breakpoint_counter = Counter(all_breakpoint_chars)
        
        # Token length statistics
        if token_lengths:
            token_length_stats = {
                'mean': np.mean(token_lengths),
                'median': np.median(token_lengths),
                'std': np.std(token_lengths),
                'min': np.min(token_lengths),
                'max': np.max(token_lengths),
                'percentiles': {
                    '25': np.percentile(token_lengths, 25),
                    '50': np.percentile(token_lengths, 50),
                    '75': np.percentile(token_lengths, 75),
                    '90': np.percentile(token_lengths, 90),
                    '95': np.percentile(token_lengths, 95),
                    '99': np.percentile(token_lengths, 99),
                }
            }
        else:
            token_length_stats = {}
        
        # Store statistics
        self.stats = {
            'total_tokens': len(all_tokens),
            'unique_tokens': len(token_counter),
            'total_smiles': len(self.results),
            'token_frequency': dict(token_counter.most_common()),
            'top_50_tokens': token_counter.most_common(50),
            'breakpoint_frequency': dict(breakpoint_counter.most_common()),
            'top_50_breakpoints': breakpoint_counter.most_common(50),
            'token_length_distribution': token_lengths,
            'token_length_stats': token_length_stats,
        }
        
        # Average tokens per SMILES
        valid_results = [r for r in self.results if 'error' not in r]
        if valid_results:
            tokens_per_smiles = [r['num_tokens'] for r in valid_results]
            self.stats['avg_tokens_per_smiles'] = np.mean(tokens_per_smiles)
            self.stats['median_tokens_per_smiles'] = np.median(tokens_per_smiles)
        else:
            self.stats['avg_tokens_per_smiles'] = 0
            self.stats['median_tokens_per_smiles'] = 0
    
    def get_top_tokens(self, n: int = 50) -> List[Tuple[str, int]]:
        """Get top n most frequent tokens."""
        return self.stats['top_50_tokens'][:n]
    
    def get_top_breakpoints(self, n: int = 50) -> List[Tuple[str, int]]:
        """Get top n most frequent breakpoint characters."""
        return self.stats['top_50_breakpoints'][:n]
    
    def get_token_length_stats(self) -> Dict:
        """Get token length statistics."""
        return self.stats['token_length_stats']
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_tokens': self.stats['total_tokens'],
            'unique_tokens': self.stats['unique_tokens'],
            'total_smiles': self.stats['total_smiles'],
            'avg_tokens_per_smiles': self.stats['avg_tokens_per_smiles'],
            'median_tokens_per_smiles': self.stats['median_tokens_per_smiles'],
            'token_length_stats': self.stats['token_length_stats'],
        }
    
    def save(self, output_path: str):
        """
        Save statistics to file.
        
        Args:
            output_path: Path to save statistics (JSON format)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        stats_copy = self.stats.copy()
        
        # Convert token_length_distribution to summary (too large for JSON)
        if 'token_length_distribution' in stats_copy:
            lengths = stats_copy['token_length_distribution']
            stats_copy['token_length_distribution_summary'] = {
                'count': len(lengths),
                'histogram': np.histogram(lengths, bins=20)[0].tolist()
            }
            del stats_copy['token_length_distribution']
        
        # Convert all numpy types to Python native types
        stats_copy = convert_to_python_types(stats_copy)
        
        with open(output_path, 'w') as f:
            json.dump(stats_copy, f, indent=2)
        
        print(f"Statistics saved to {output_path}")
    
    @classmethod
    def load(cls, input_path: str) -> 'TokenStatistics':
        """
        Load statistics from file.
        
        Args:
            input_path: Path to statistics file
        
        Returns:
            TokenStatistics instance
        """
        with open(input_path, 'r') as f:
            stats = json.load(f)
        
        # Create empty instance and set stats directly
        instance = cls.__new__(cls)
        instance.results = []
        instance.stats = stats
        return instance


def compute_token_statistics(tokenization_results: List[Dict]) -> TokenStatistics:
    """
    Compute statistics from tokenization results.
    
    Args:
        tokenization_results: List of tokenization dictionaries
    
    Returns:
        TokenStatistics instance
    """
    return TokenStatistics(tokenization_results)


def compare_token_distributions(stats1: TokenStatistics, stats2: TokenStatistics,
                                label1: str = "Model 1", label2: str = "Model 2") -> Dict:
    """
    Compare two token distributions.
    
    Args:
        stats1: First TokenStatistics instance
        stats2: Second TokenStatistics instance
        label1: Label for first distribution
        label2: Label for second distribution
    
    Returns:
        Dictionary with comparison metrics
    """
    # Get top tokens from both
    tokens1 = set([t for t, _ in stats1.get_top_tokens(100)])
    tokens2 = set([t for t, _ in stats2.get_top_tokens(100)])
    
    # Compute overlap
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    jaccard_similarity = len(intersection) / len(union) if union else 0
    overlap_count = len(intersection)
    
    # Compare token length statistics
    length_stats1 = stats1.get_token_length_stats()
    length_stats2 = stats2.get_token_length_stats()
    
    length_diff = {
        'mean_diff': length_stats2.get('mean', 0) - length_stats1.get('mean', 0),
        'median_diff': length_stats2.get('median', 0) - length_stats1.get('median', 0),
    }
    
    # Get breakpoint comparison
    bp1 = set([bp for bp, _ in stats1.get_top_breakpoints(50)])
    bp2 = set([bp for bp, _ in stats2.get_top_breakpoints(50)])
    
    bp_intersection = bp1 & bp2
    bp_union = bp1 | bp2
    bp_jaccard = len(bp_intersection) / len(bp_union) if bp_union else 0
    
    return {
        'label1': label1,
        'label2': label2,
        'token_overlap': {
            'intersection_count': overlap_count,
            'jaccard_similarity': jaccard_similarity,
            'unique_to_1': len(tokens1 - tokens2),
            'unique_to_2': len(tokens2 - tokens1),
        },
        'length_comparison': length_diff,
        'breakpoint_overlap': {
            'intersection_count': len(bp_intersection),
            'jaccard_similarity': bp_jaccard,
            'unique_to_1': len(bp1 - bp2),
            'unique_to_2': len(bp2 - bp1),
        },
        'summary_1': stats1.get_summary(),
        'summary_2': stats2.get_summary(),
    }


def compute_kl_divergence(stats1: TokenStatistics, stats2: TokenStatistics,
                         top_n: int = 1000) -> float:
    """
    Compute KL divergence between two token distributions.
    
    Args:
        stats1: First TokenStatistics instance
        stats2: Second TokenStatistics instance
        top_n: Number of top tokens to consider
    
    Returns:
        KL divergence value
    """
    # Get token frequencies
    freq1 = dict(stats1.stats['token_frequency'])
    freq2 = dict(stats2.stats['token_frequency'])
    
    # Get union of tokens
    all_tokens = set(list(freq1.keys())[:top_n]) | set(list(freq2.keys())[:top_n])
    
    # Convert to probabilities
    total1 = sum(freq1.values())
    total2 = sum(freq2.values())
    
    prob1 = np.array([freq1.get(t, 1e-10) / total1 for t in all_tokens])
    prob2 = np.array([freq2.get(t, 1e-10) / total2 for t in all_tokens])
    
    # Add smoothing
    prob1 = prob1 + 1e-10
    prob2 = prob2 + 1e-10
    
    # Normalize
    prob1 = prob1 / prob1.sum()
    prob2 = prob2 / prob2.sum()
    
    # Compute KL divergence
    kl_div = np.sum(prob1 * np.log(prob1 / prob2))
    
    return kl_div


def create_comparison_dataframe(statistics_dict: Dict[str, TokenStatistics]) -> pd.DataFrame:
    """
    Create a comparison DataFrame from multiple statistics.
    
    Args:
        statistics_dict: Dictionary mapping model names to TokenStatistics
    
    Returns:
        DataFrame with comparison metrics
    """
    rows = []
    
    for name, stats in statistics_dict.items():
        summary = stats.get_summary()
        length_stats = summary['token_length_stats']
        
        row = {
            'Model': name,
            'Total Tokens': summary['total_tokens'],
            'Unique Tokens': summary['unique_tokens'],
            'Total SMILES': summary['total_smiles'],
            'Avg Tokens/SMILES': f"{summary['avg_tokens_per_smiles']:.2f}",
            'Median Tokens/SMILES': f"{summary['median_tokens_per_smiles']:.2f}",
            'Mean Token Length': f"{length_stats.get('mean', 0):.2f}",
            'Median Token Length': f"{length_stats.get('median', 0):.2f}",
            'Std Token Length': f"{length_stats.get('std', 0):.2f}",
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

