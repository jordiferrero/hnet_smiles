"""
SMI-TED Featurizer: Extract embeddings from IBM's SMI-TED transformer model.

Reference: https://github.com/IBM/materials
Paper: "SMI-TED: A Foundation Model for Molecular Property Prediction"

SMI-TED is a pre-trained transformer for SMILES that can be used for:
- Frozen inference (feature extraction)
- Fine-tuning on downstream tasks
"""

import numpy as np
from typing import List, Optional
from pathlib import Path
import warnings

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Install with: pip install transformers torch")


class SMITEDFeaturizer:
    """
    Extract features from SMI-TED transformer model.
    
    Uses frozen inference to get embeddings from the pre-trained model.
    """
    
    # Model identifiers
    MODEL_NAME = "ibm/smi-ted"  # Hugging Face model ID
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        pooling: str = 'mean',
    ):
        """
        Initialize SMI-TED featurizer.
        
        Args:
            model_name: Model name/path (default: ibm/smi-ted)
            device: Device to run on ('cuda' or 'cpu')
            pooling: Pooling strategy - 'mean', 'cls', or 'max'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required. Install with: pip install transformers")
        
        self.model_name = model_name or self.MODEL_NAME
        self.pooling = pooling
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.tokenizer = None
        self.model = None
        self.feature_dim = 768  # SMI-TED hidden size (typical transformer)
        
        # Lazy loading
        self._loaded = False
    
    def _load_model(self):
        """Load the SMI-TED model."""
        if self._loaded:
            return
        
        try:
            print(f"Loading SMI-TED model from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get actual hidden size
            self.feature_dim = self.model.config.hidden_size
            
            self._loaded = True
            print(f"SMI-TED loaded on {self.device}, feature_dim={self.feature_dim}")
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SMI-TED model: {e}\n"
                "The model may not be publicly available yet.\n"
                "See https://github.com/IBM/materials for setup instructions."
            )
    
    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pooling to hidden states."""
        if self.pooling == 'cls':
            return hidden_states[:, 0, :]
        
        elif self.pooling == 'mean':
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            return summed / lengths
        
        elif self.pooling == 'max':
            # Masked max pooling
            mask = attention_mask.unsqueeze(-1).bool()
            hidden_states = hidden_states.masked_fill(~mask, float('-inf'))
            return hidden_states.max(dim=1).values
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def featurize_single(self, smiles: str) -> np.ndarray:
        """
        Extract features for a single SMILES.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Feature vector of shape (D,)
        """
        self._load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            smiles,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            pooled = self._pool(hidden_states, inputs['attention_mask'])
        
        return pooled.squeeze(0).cpu().numpy()
    
    def featurize_batch(self, smiles_list: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Extract features for a list of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            show_progress: Show progress bar
            
        Returns:
            Feature matrix of shape (N, D)
        """
        self._load_model()
        
        from tqdm import tqdm
        
        features = []
        iterator = tqdm(smiles_list, desc="SMI-TED features") if show_progress else smiles_list
        
        for smiles in iterator:
            try:
                feat = self.featurize_single(smiles)
                features.append(feat)
            except Exception as e:
                print(f"Error processing '{smiles}': {e}")
                features.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        return np.array(features)
    
    @property
    def name(self) -> str:
        return f"SMI-TED_{self.pooling}"


def setup_smited():
    """
    Setup instructions for SMI-TED.
    """
    instructions = """
    ============================================================
    SMI-TED Setup Instructions
    ============================================================
    
    1. Install dependencies:
       pip install transformers torch
    
    2. The model should be available on Hugging Face:
       from transformers import AutoModel, AutoTokenizer
       model = AutoModel.from_pretrained("ibm/smi-ted")
       tokenizer = AutoTokenizer.from_pretrained("ibm/smi-ted")
    
    3. Alternatively, use the IBM materials repository:
       git clone https://github.com/IBM/materials.git external/materials
       
       See notebooks for usage:
       - smi_ted_frozen_inference_example1.ipynb (BBBP)
       - smi_ted_frozen_inference_example2.ipynb (Lipophilicity)
    
    4. For frozen inference, use the featurize_batch method.
       For fine-tuning, use the SMI-TED training scripts directly.
    ============================================================
    """
    print(instructions)


if __name__ == '__main__':
    setup_smited()





