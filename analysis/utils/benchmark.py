"""
Utilities for SmilesPE benchmark tokenization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class SmilesPEBenchmark:
    """
    Wrapper for SmilesPE tokenizer benchmark.
    """
    
    def __init__(self, vocab_path: str = None):
        """
        Initialize SmilesPE tokenizer.
        
        Args:
            vocab_path: Path to SPE vocabulary file (SPE_ChEMBL.txt)
                       If None, will look in analysis/data/ directory
        """
        try:
            import codecs
            from SmilesPE.tokenizer import SPE_Tokenizer
            self.SPE_Tokenizer = SPE_Tokenizer
            self.codecs = codecs
        except ImportError:
            raise ImportError(
                "SmilesPE not installed. Install with: pip install SmilesPE"
            )
        
        # Set vocab path
        if vocab_path is None:
            vocab_path = str(project_root / 'analysis' / 'data' / 'SPE_ChEMBL.txt')
        
        self.vocab_path = vocab_path
        
        if not Path(vocab_path).exists():
            raise FileNotFoundError(
                f"SPE vocabulary file not found at {vocab_path}. "
                f"Please download SPE_ChEMBL.txt from SmilesPE GitHub repo."
            )
        
        # Load tokenizer
        self.tokenizer = self.SPE_Tokenizer(
            self.codecs.open(vocab_path, 'r', 'utf-8')
        )
        print(f"SmilesPE tokenizer loaded from {vocab_path}")
    
    def tokenize_smiles(self, smiles: str) -> List[str]:
        """
        Tokenize a single SMILES string.
        
        Args:
            smiles: SMILES string
        
        Returns:
            List of token strings
        """
        tokenized = self.tokenizer.tokenize(smiles)
        tokens = tokenized.split(' ')
        return tokens
    
    def tokenize_dataset(self, dataset_csv: str, dataset_type: str,
                        max_samples: Optional[int] = None) -> List[Dict]:
        """
        Tokenize entire dataset with SmilesPE.
        
        Args:
            dataset_csv: Path to dataset CSV file
            dataset_type: 'PI1M' or 'MOSES'
            max_samples: Maximum number of samples to process (None = all)
        
        Returns:
            List of tokenization results (same format as H-Net inference)
        """
        # Load dataset
        print(f"Loading dataset from {dataset_csv}...")
        df = pd.read_csv(dataset_csv)
        
        # Get SMILES column
        if dataset_type == 'PI1M':
            smiles_col = 'SMILES'
        elif dataset_type == 'MOSES':
            smiles_col = 'smiles'
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        smiles_list = df[smiles_col].dropna().tolist()
        
        if max_samples:
            smiles_list = smiles_list[:max_samples]
        
        print(f"Tokenizing {len(smiles_list)} SMILES strings with SmilesPE...")
        
        results = []
        for smiles in tqdm(smiles_list, desc="SmilesPE tokenizing"):
            try:
                tokens = self.tokenize_smiles(smiles)
                
                # Calculate breakpoints (indices where tokens end)
                breakpoints = []
                breakpoint_chars = []
                current_pos = 0
                
                for token in tokens[:-1]:  # Exclude last token
                    current_pos += len(token)
                    breakpoints.append(current_pos)
                    if current_pos < len(smiles):
                        breakpoint_chars.append(smiles[current_pos - 1])
                
                result = {
                    'text': smiles,
                    'tokens': tokens,
                    'breakpoints': breakpoints,
                    'breakpoint_chars': breakpoint_chars,
                    'num_tokens': len(tokens),
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error processing SMILES '{smiles}': {e}")
                results.append({
                    'text': smiles,
                    'tokens': [],
                    'breakpoints': [],
                    'breakpoint_chars': [],
                    'num_tokens': 0,
                    'error': str(e)
                })
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save tokenization results.
        
        Args:
            results: List of tokenization results
            output_path: Path to save results
        """
        from .inference import save_tokenization_results
        save_tokenization_results(results, output_path)
    
    @staticmethod
    def load_results(input_path: str) -> List[Dict]:
        """
        Load tokenization results.
        
        Args:
            input_path: Path to results file
        
        Returns:
            List of tokenization results
        """
        from .inference import load_tokenization_results
        return load_tokenization_results(input_path)


def download_spe_vocabulary(output_dir: str = None):
    """
    Download SPE_ChEMBL.txt vocabulary file from SmilesPE GitHub.
    
    Args:
        output_dir: Directory to save vocabulary file
                   If None, saves to analysis/data/
    """
    import urllib.request
    
    if output_dir is None:
        output_dir = str(project_root / 'analysis' / 'data')
    
    output_path = Path(output_dir) / 'SPE_ChEMBL.txt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    url = "https://raw.githubusercontent.com/XinhaoLi74/SmilesPE/master/SPE_ChEMBL.txt"
    
    print(f"Downloading SPE vocabulary from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Vocabulary saved to {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Error downloading vocabulary: {e}")
        print("Please manually download SPE_ChEMBL.txt from:")
        print("https://github.com/XinhaoLi74/SmilesPE/blob/master/SPE_ChEMBL.txt")
        return None


def setup_smilespE():
    """
    Setup SmilesPE: install package and download vocabulary.
    
    Returns:
        Path to vocabulary file if successful, None otherwise
    """
    import subprocess
    
    # Check if SmilesPE is installed
    try:
        import SmilesPE
        print("SmilesPE is already installed.")
    except ImportError:
        print("Installing SmilesPE...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "SmilesPE"
            ], check=True)
            print("SmilesPE installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing SmilesPE: {e}")
            return None
    
    # Download vocabulary
    vocab_path = download_spe_vocabulary()
    return vocab_path

