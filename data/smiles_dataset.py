"""
SMILES Dataset class for HNet training.
Handles loading, tokenization, and batching of SMILES strings.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import random

from hnet.utils.tokenizers import ByteTokenizer


class SMILESDataset(Dataset):
    """
    Dataset for SMILES strings with support for concatenation strategies.
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer: ByteTokenizer,
        max_samples: Optional[int] = None,
        concatenate: bool = True,
        num_concatenate: int = 3,
        separator: str = " ",
        max_seq_length: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize SMILES dataset.
        
        Args:
            csv_path: Path to CSV file with SMILES data
            tokenizer: ByteTokenizer instance
            max_samples: Maximum number of samples to use (None = all)
            concatenate: Whether to concatenate multiple SMILES
            num_concatenate: Number of SMILES to concatenate (if concatenate=True)
            separator: Separator between concatenated SMILES
            max_seq_length: Maximum sequence length (for filtering)
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.concatenate = concatenate
        self.num_concatenate = num_concatenate
        self.separator = separator
        self.max_seq_length = max_seq_length
        self.seed = seed
        
        # Load SMILES from CSV
        print(f"Loading SMILES from {csv_path}...")
        df = pd.read_csv(csv_path)
        smiles_col = df.columns[0]
        smiles = df[smiles_col].astype(str).tolist()
        
        if max_samples:
            smiles = smiles[:max_samples]
        
        print(f"Loaded {len(smiles)} SMILES strings")
        
        # Process SMILES based on strategy
        if concatenate:
            self.smiles = self._concatenate_smiles(smiles)
        else:
            self.smiles = smiles
        
        # Filter by max length if specified
        if max_seq_length:
            original_len = len(self.smiles)
            self.smiles = [
                s for s in self.smiles 
                if len(s) <= max_seq_length
            ]
            print(f"Filtered to {len(self.smiles)} sequences (removed {original_len - len(self.smiles)} longer than {max_seq_length})")
        
        print(f"Final dataset size: {len(self.smiles)} sequences")
        
        # Compute statistics
        lengths = [len(s) for s in self.smiles]
        if lengths:
            print(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}")
    
    def _concatenate_smiles(self, smiles: List[str]) -> List[str]:
        """Concatenate multiple SMILES strings together."""
        random.seed(self.seed)
        concatenated = []
        
        # Shuffle for random concatenation
        shuffled = smiles.copy()
        random.shuffle(shuffled)
        
        # Group into batches for concatenation
        for i in range(0, len(shuffled), self.num_concatenate):
            batch = shuffled[i:i + self.num_concatenate]
            if len(batch) == self.num_concatenate:
                concatenated.append(self.separator.join(batch))
        
        return concatenated
    
    def __len__(self) -> int:
        return len(self.smiles)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single SMILES sequence."""
        smiles_str = self.smiles[idx]
        
        # Tokenize
        encoded = self.tokenizer.encode(
            [smiles_str],
            add_bos=True,
            add_eos=True
        )[0]
        
        return {
            'input_ids': encoded['input_ids'],
            'text': smiles_str,
        }


def collate_fn_packed(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for packed sequences (for efficient batching).
    Returns sequences in packed format with cu_seqlens.
    """
    # Extract input_ids
    input_ids_list = [item['input_ids'] for item in batch]
    
    # Flatten all sequences
    flat_input_ids = np.concatenate(input_ids_list)
    
    # Create cu_seqlens (cumulative sequence lengths)
    cu_seqlens = np.zeros(len(batch) + 1, dtype=np.int32)
    for i, ids in enumerate(input_ids_list):
        cu_seqlens[i + 1] = cu_seqlens[i] + len(ids)
    
    # Convert to tensors
    flat_input_ids = torch.tensor(flat_input_ids, dtype=torch.long)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
    
    # Create labels (shifted by 1 for next-token prediction)
    labels = torch.cat([
        flat_input_ids[cu_seqlens[i]:cu_seqlens[i+1]][1:]
        for i in range(len(batch))
    ])
    
    # Inputs are all tokens except last
    inputs = flat_input_ids[:-1]
    
    return {
        'input_ids': inputs,
        'labels': labels,
        'cu_seqlens': cu_seqlens,
        'batch_size': len(batch),
    }


def collate_fn_padded(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for padded sequences (alternative to packed).
    """
    # Find max length
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Pad sequences
    input_ids = []
    masks = []
    for item in batch:
        ids = item['input_ids']
        padding = max_len - len(ids)
        padded_ids = np.pad(ids, (0, padding), constant_values=0)
        mask = np.ones(len(ids), dtype=bool)
        mask = np.pad(mask, (0, padding), constant_values=False)
        
        input_ids.append(padded_ids)
        masks.append(mask)
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.bool)
    
    # Create labels (shifted by 1)
    labels = input_ids[:, 1:].clone()
    input_ids = input_ids[:, :-1]
    masks = masks[:, :-1]
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'mask': masks,
        'batch_size': len(batch),
    }


def create_dataloader(
    csv_path: str,
    tokenizer: ByteTokenizer,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    concatenate: bool = True,
    num_concatenate: int = 3,
    use_packed: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for SMILES data.
    
    Args:
        csv_path: Path to CSV file
        tokenizer: ByteTokenizer instance
        batch_size: Batch size
        max_samples: Maximum samples to use
        concatenate: Whether to concatenate SMILES
        num_concatenate: Number to concatenate
        use_packed: Use packed sequences (more efficient) vs padded
        shuffle: Shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for SMILESDataset
    
    Returns:
        DataLoader instance
    """
    dataset = SMILESDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_samples=max_samples,
        concatenate=concatenate,
        num_concatenate=num_concatenate,
        **kwargs
    )
    
    collate_fn = collate_fn_packed if use_packed else collate_fn_padded
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == '__main__':
    # Test the dataset
    tokenizer = ByteTokenizer()
    
    csv_path = '../datasets/PI1M/PI1M_v2.csv'
    
    print("Testing SMILESDataset with concatenation...")
    dataset = SMILESDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_samples=100,
        concatenate=True,
        num_concatenate=3,
    )
    
    print(f"\nDataset length: {len(dataset)}")
    print(f"\nSample sequences:")
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        print(f"  {i}: length={len(item['input_ids'])}, text preview={item['text'][:50]}...")
    
    print("\nTesting DataLoader...")
    dataloader = create_dataloader(
        csv_path=csv_path,
        tokenizer=tokenizer,
        batch_size=4,
        max_samples=100,
        concatenate=True,
        num_concatenate=3,
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    if 'cu_seqlens' in batch:
        print(f"cu_seqlens: {batch['cu_seqlens']}")

