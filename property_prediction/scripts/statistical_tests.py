#!/usr/bin/env python3
"""
Statistical Significance Tests for Property Prediction Results.

Computes:
1. Bootstrap confidence intervals
2. Paired t-tests (H-Net vs RDKit)
3. Cohen's d effect sizes
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Paths
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'tables'


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Args:
        scores: Array of scores (e.g., from CV folds)
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence level (default 95%)
    
    Returns:
        (lower, upper) confidence interval bounds
    """
    bootstrapped = []
    n = len(scores)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrapped.append(np.mean(sample))
    
    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    
    lower = np.percentile(bootstrapped, lower_percentile)
    upper = np.percentile(bootstrapped, upper_percentile)
    
    return lower, upper


def paired_ttest(scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
    """
    Perform paired t-test for dependent samples.
    
    Args:
        scores1: First set of scores (e.g., H-Net)
        scores2: Second set of scores (e.g., RDKit)
    
    Returns:
        (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    return t_stat, p_value


def cohens_d(mean1: float, mean2: float, std1: float, std2: float) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        mean1, mean2: Means of two groups
        std1, std2: Standard deviations of two groups
    
    Returns:
        Cohen's d (positive means group1 > group2)
    """
    # Pooled standard deviation
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def compute_significance(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical significance metrics for all results.
    """
    enhanced_results = []
    
    for _, row in results_df.iterrows():
        result = row.to_dict()
        
        # Get fold scores if available
        fold_scores = row.get('fold_scores', None)
        if isinstance(fold_scores, str):
            # Parse string representation of list
            try:
                fold_scores = eval(fold_scores)
            except:
                fold_scores = None
        
        if fold_scores and isinstance(fold_scores, (list, np.ndarray)):
            fold_scores = np.array(fold_scores)
            
            # Bootstrap CI
            lower, upper = bootstrap_ci(fold_scores)
            result['ci_lower'] = lower
            result['ci_upper'] = upper
            result['ci_width'] = upper - lower
        
        enhanced_results.append(result)
    
    return pd.DataFrame(enhanced_results)


def load_existing_results():
    """Load all existing results."""
    all_results = []
    
    # Load existing results from tables
    files = [
        'all_results.csv',
        'moleculenet_extended_results.csv',
        'bbbp_results.csv',
        'lipophilicity_results.csv',
        'tg_results.csv',
        'mac_results.csv',
    ]
    
    for filename in files:
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"Loaded {filepath}: {len(df)} rows")
            all_results.append(df)
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        return combined
    
    return pd.DataFrame()


def main():
    """Main function to compute statistical tests."""
    print("Loading results...")
    results = load_existing_results()
    
    if results.empty:
        print("No results found.")
        return
    
    print(f"\nTotal results: {len(results)} rows")
    
    # Add bootstrap CIs
    print("\nComputing bootstrap confidence intervals...")
    enhanced = compute_significance(results)
    
    # Save enhanced results
    output_path = RESULTS_DIR / 'results_with_statistics.csv'
    enhanced.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Print summary
    print("\n=== Results Summary with 95% CI ===")
    
    for task in ['regression', 'classification']:
        task_results = enhanced[enhanced['task'] == task] if 'task' in enhanced.columns else enhanced
        if task_results.empty:
            continue
        
        print(f"\n{task.upper()} Tasks:")
        
        for _, row in task_results.iterrows():
            model = row.get('model', row.get('Model', 'Unknown'))
            dataset = row.get('dataset', row.get('Dataset', 'Unknown'))
            
            if task == 'regression' and 'rmse' in row:
                metric = row['rmse']
                ci_lower = row.get('ci_lower', metric)
                ci_upper = row.get('ci_upper', metric)
                print(f"  {dataset} ({model}): RMSE = {metric:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
            elif task == 'classification' and 'auc' in row:
                metric = row['auc']
                ci_lower = row.get('ci_lower', metric)
                ci_upper = row.get('ci_upper', metric)
                print(f"  {dataset} ({model}): AUC = {metric:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return enhanced


if __name__ == '__main__':
    results = main()




