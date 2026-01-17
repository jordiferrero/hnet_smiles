"""
Dataset loaders for property prediction.

Provides unified interface for loading:
- PI1M_Tg_MAC (polymer properties)
- Lipophilicity (molecule property)
- BBBP (molecule property)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Literal
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent / 'data'


class PropertyDataset:
    """Base class for property prediction datasets."""
    
    def __init__(
        self,
        smiles: np.ndarray,
        targets: np.ndarray,
        names: Optional[np.ndarray] = None,
    ):
        self.smiles = smiles
        self.targets = targets
        self.names = names
    
    def __len__(self) -> int:
        return len(self.smiles)
    
    def train_test_split(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = False,
    ) -> Tuple['PropertyDataset', 'PropertyDataset', 'PropertyDataset']:
        """
        Split into train/val/test sets.
        
        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            stratify: Whether to stratify (for classification)
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        stratify_labels = self.targets if stratify else None
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            np.arange(len(self)),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
        
        # Second split: train vs val
        remaining_stratify = self.targets[train_val_idx] if stratify else None
        val_ratio = val_size / (1 - test_size)
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            random_state=random_state,
            stratify=remaining_stratify,
        )
        
        def make_subset(indices):
            return PropertyDataset(
                smiles=self.smiles[indices],
                targets=self.targets[indices],
                names=self.names[indices] if self.names is not None else None,
            )
        
        return make_subset(train_idx), make_subset(val_idx), make_subset(test_idx)


def load_pi1m_tg_mac(
    target: Literal['Tg', 'MAC'] = 'Tg',
    max_samples: Optional[int] = None,
) -> PropertyDataset:
    """
    Load PI1M_Tg_MAC dataset for polymer property prediction.
    
    Args:
        target: Target property - 'Tg' or 'MAC'
        max_samples: Maximum number of samples (None = all)
        
    Returns:
        PropertyDataset with SMILES and targets
        
    Note:
        The dataset contains:
        - Tg: Glass transition temperature (°C), experimental or surrogate-predicted
        - Tg_predicted: Binary flag (0=experimental, 1=surrogate-predicted)
        - MAC: Mass attenuation coefficient
    """
    csv_path = DATA_DIR / 'polymer' / 'PI1M_Tg_MAC.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PI1M_Tg_MAC.csv not found at {csv_path}. "
            "Please ensure the dataset is in the correct location."
        )
    
    print(f"Loading PI1M_Tg_MAC dataset...")
    df = pd.read_csv(csv_path)
    
    if max_samples is not None:
        df = df.head(max_samples)
    
    # Get SMILES
    smiles = df['SMILES'].values
    names = df['Name'].values if 'Name' in df.columns else None
    
    # Get target - use actual Tg values (not the binary flag Tg_predicted)
    if target == 'Tg':
        targets = df['Tg'].values  # Actual Tg in °C
    elif target == 'MAC':
        targets = df['MAC'].values
    else:
        raise ValueError(f"Unknown target: {target}. Choose 'Tg' or 'MAC'")
    
    # Remove NaN targets
    valid_mask = ~np.isnan(targets)
    smiles = smiles[valid_mask]
    targets = targets[valid_mask]
    if names is not None:
        names = names[valid_mask]
    
    print(f"  Loaded {len(smiles)} samples")
    print(f"  Target ({target}) range: [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"  Target mean: {targets.mean():.2f}, std: {targets.std():.2f}")
    
    return PropertyDataset(smiles=smiles, targets=targets, names=names)


def load_lipophilicity(
    max_samples: Optional[int] = None,
) -> PropertyDataset:
    """
    Load Lipophilicity dataset from MoleculeNet.
    
    Args:
        max_samples: Maximum number of samples
        
    Returns:
        PropertyDataset with SMILES and logD targets
    """
    csv_path = DATA_DIR / 'molecule' / 'lipophilicity.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"lipophilicity.csv not found at {csv_path}. "
            "Run: python scripts/download_datasets.py"
        )
    
    print(f"Loading Lipophilicity dataset...")
    df = pd.read_csv(csv_path)
    
    if max_samples is not None:
        df = df.head(max_samples)
    
    # Column names may vary
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    target_col = 'exp' if 'exp' in df.columns else 'Lipophilicity'
    
    smiles = df[smiles_col].values
    targets = df[target_col].values
    
    # Remove NaN
    valid_mask = ~pd.isna(targets) & ~pd.isna(smiles)
    smiles = smiles[valid_mask]
    targets = targets[valid_mask].astype(np.float32)
    
    print(f"  Loaded {len(smiles)} samples")
    print(f"  Target range: [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"  Target mean: {targets.mean():.2f}, std: {targets.std():.2f}")
    
    return PropertyDataset(smiles=smiles, targets=targets)


def load_bbbp(
    max_samples: Optional[int] = None,
) -> PropertyDataset:
    """
    Load BBBP dataset from MoleculeNet.
    
    BBBP: Blood-Brain Barrier Penetration (binary classification)
    
    Args:
        max_samples: Maximum number of samples
        
    Returns:
        PropertyDataset with SMILES and binary targets
    """
    csv_path = DATA_DIR / 'molecule' / 'bbbp.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"bbbp.csv not found at {csv_path}. "
            "Run: python scripts/download_datasets.py"
        )
    
    print(f"Loading BBBP dataset...")
    df = pd.read_csv(csv_path)
    
    if max_samples is not None:
        df = df.head(max_samples)
    
    # Column names
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    target_col = 'p_np' if 'p_np' in df.columns else 'BBBP'
    
    smiles = df[smiles_col].values
    targets = df[target_col].values
    
    # Remove NaN
    valid_mask = ~pd.isna(targets) & ~pd.isna(smiles)
    smiles = smiles[valid_mask]
    targets = targets[valid_mask].astype(np.int32)
    
    print(f"  Loaded {len(smiles)} samples")
    print(f"  Class distribution: {dict(zip(*np.unique(targets, return_counts=True)))}")
    
    return PropertyDataset(smiles=smiles, targets=targets)


def get_dataset_info() -> dict:
    """Get information about available datasets."""
    return {
        'PI1M_Tg_MAC': {
            'path': str(DATA_DIR / 'polymer' / 'PI1M_Tg_MAC.csv'),
            'exists': (DATA_DIR / 'polymer' / 'PI1M_Tg_MAC.csv').exists(),
            'targets': ['Tg', 'MAC'],
            'task': 'regression',
            'description': 'Polymer glass transition temperature and mass attenuation coefficient',
        },
        'Lipophilicity': {
            'path': str(DATA_DIR / 'molecule' / 'lipophilicity.csv'),
            'exists': (DATA_DIR / 'molecule' / 'lipophilicity.csv').exists(),
            'targets': ['exp'],
            'task': 'regression',
            'description': 'Octanol/water distribution coefficient (logD)',
        },
        'BBBP': {
            'path': str(DATA_DIR / 'molecule' / 'bbbp.csv'),
            'exists': (DATA_DIR / 'molecule' / 'bbbp.csv').exists(),
            'targets': ['p_np'],
            'task': 'classification',
            'description': 'Blood-brain barrier penetration (binary)',
        },
    }


if __name__ == '__main__':
    # Test dataset loading
    print("Dataset Information:")
    print("=" * 60)
    for name, info in get_dataset_info().items():
        status = "✓" if info['exists'] else "✗"
        print(f"{status} {name}: {info['description']}")
        print(f"  Path: {info['path']}")
        print(f"  Task: {info['task']}")
        print()


