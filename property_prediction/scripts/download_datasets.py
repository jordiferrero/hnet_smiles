#!/usr/bin/env python3
"""
Download MoleculeNet datasets for property prediction.

Downloads:
- Lipophilicity (regression)
- BBBP (classification)
"""

import os
import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
import tempfile


# Data directory
DATA_DIR = Path(__file__).parent.parent / 'data' / 'molecule'

# MoleculeNet dataset URLs (from DeepChem)
DATASETS = {
    'lipophilicity': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
        'filename': 'lipophilicity.csv',
        'smiles_col': 'smiles',
        'target_col': 'exp',
        'task': 'regression',
    },
    'bbbp': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
        'filename': 'bbbp.csv',
        'smiles_col': 'smiles',
        'target_col': 'p_np',
        'task': 'classification',
    },
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL."""
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"  Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def download_lipophilicity() -> bool:
    """Download Lipophilicity dataset."""
    info = DATASETS['lipophilicity']
    dest_path = DATA_DIR / info['filename']
    
    if dest_path.exists():
        print(f"Lipophilicity already exists at {dest_path}")
        df = pd.read_csv(dest_path)
        print(f"  {len(df)} samples")
        return True
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    success = download_file(info['url'], dest_path)
    
    if success:
        # Verify and show stats
        df = pd.read_csv(dest_path)
        print(f"  {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Target stats: mean={df[info['target_col']].mean():.3f}, std={df[info['target_col']].std():.3f}")
    
    return success


def download_bbbp() -> bool:
    """Download BBBP dataset."""
    info = DATASETS['bbbp']
    dest_path = DATA_DIR / info['filename']
    
    if dest_path.exists():
        print(f"BBBP already exists at {dest_path}")
        df = pd.read_csv(dest_path)
        print(f"  {len(df)} samples")
        return True
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    success = download_file(info['url'], dest_path)
    
    if success:
        # Verify and show stats
        df = pd.read_csv(dest_path)
        print(f"  {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Class distribution: {df[info['target_col']].value_counts().to_dict()}")
    
    return success


def download_all() -> bool:
    """Download all datasets."""
    print("=" * 60)
    print("Downloading MoleculeNet datasets")
    print("=" * 60)
    
    success = True
    
    print("\n1. Lipophilicity (regression)")
    success &= download_lipophilicity()
    
    print("\n2. BBBP (classification)")
    success &= download_bbbp()
    
    print("\n" + "=" * 60)
    if success:
        print("All datasets downloaded successfully!")
    else:
        print("Some downloads failed. Check errors above.")
    print("=" * 60)
    
    return success


if __name__ == '__main__':
    download_all()




