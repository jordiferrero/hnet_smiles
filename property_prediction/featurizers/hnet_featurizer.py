"""
H-Net Featurizer: Extract latent representations from trained H-Net models.

Supports 3 pooling strategies:
- mean: Mean pooling over sequence length
- max: Max pooling over sequence length  
- cls: First token (CLS-like) representation
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Literal
from tqdm import tqdm
import sys

# Add project root for H-Net imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.utils.tokenizers import ByteTokenizer


# Available H-Net checkpoints
POLYMER_CHECKPOINTS = {
    'hnet_1ep_nocat_1stg': 'run_large_20251107_133202',
    'hnet_5ep_nocat_1stg': 'run_large_20251111_075600',
    'hnet_5ep_cat10_1stg': 'run_large_20251111_181836',
    'hnet_22ep_cat10_1stg': 'run_large_20251112_150502',
    'hnet_5ep_cat10_2stg': 'run_large_20260115_191350',
}

MOLECULE_CHECKPOINTS = {
    'hnet_5ep_cat10_1stg': 'run_large_20251112_071557',
    'hnet_5ep_nocat_1stg': 'run_large_20251113_074900',
    'hnet_5ep_cat10_2stg': 'run_large_20260116_074355',
}


PoolingStrategy = Literal['mean', 'max', 'cls']


class HNetFeaturizer:
    """
    Extract latent representations from frozen H-Net models.
    
    The H-Net backbone produces hidden states of shape (B, L, D) where D=512.
    We apply pooling to get a fixed-size feature vector (B, D).
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        pooling: PoolingStrategy = 'mean',
        device: str = None,
        batch_size: int = 32,
    ):
        """
        Initialize H-Net featurizer.
        
        Args:
            checkpoint_dir: Path to checkpoint directory (e.g., checkpoints/run_large_20251112_150502)
            pooling: Pooling strategy - 'mean', 'max', or 'cls'
            device: Device to run on ('cuda' or 'cpu'). Auto-detect if None.
            batch_size: Batch size for feature extraction
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.pooling = pooling
        self.batch_size = batch_size
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self.model, self.model_info = self._load_model()
        self.tokenizer = ByteTokenizer()
        
        # Feature dimension (from H-Net config)
        self.feature_dim = self.model_info['config'].get('d_model', [512])[0]
    
    def _load_model(self) -> tuple:
        """Load H-Net model from checkpoint."""
        import json
        from omegaconf import ListConfig
        
        metadata_path = self.checkpoint_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata.json found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        config = metadata.get('config', {})
        
        # Create config objects
        attn_cfg = AttnConfig(**config['attn_cfg'])
        ssm_cfg = SSMConfig(**config['ssm_cfg'])
        
        config_copy = config.copy()
        config_copy.pop('attn_cfg')
        config_copy.pop('ssm_cfg')
        
        hnet_cfg = HNetConfig(**config_copy, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
        
        # Create model
        model = HNetForCausalLM(hnet_cfg, device=self.device, dtype=torch.bfloat16)
        model.eval()
        
        # Load checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoints' / 'checkpoint_bytes_best.pt'
        if not checkpoint_path.exists():
            # Try epoch checkpoint
            epoch_checkpoints = list((self.checkpoint_dir / 'checkpoints').glob('checkpoint_epoch_*.pt'))
            if epoch_checkpoints:
                checkpoint_path = sorted(epoch_checkpoints)[-1]  # Latest epoch
            else:
                raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir / 'checkpoints'}")
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        # Handle torch version compatibility
        major, minor = map(int, torch.__version__.split('.')[:2])
        if (major, minor) >= (2, 6):
            with torch.serialization.safe_globals([ListConfig]):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        print(f"Model loaded on {self.device}")
        
        # Freeze model
        for param in model.parameters():
            param.requires_grad = False
        
        model_info = {
            'run_name': metadata.get('run_name', self.checkpoint_dir.name),
            'config': config,
            'checkpoint_path': str(checkpoint_path),
        }
        
        return model, model_info
    
    def _extract_hidden_states(self, smiles: str) -> torch.Tensor:
        """
        Extract hidden states from H-Net backbone (before lm_head).
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Hidden states tensor of shape (1, L, D)
        """
        # Tokenize
        encoded = self.tokenizer.encode([smiles], add_bos=True, add_eos=True)[0]
        input_ids = torch.tensor(
            encoded['input_ids'], 
            dtype=torch.long, 
            device=self.device
        ).unsqueeze(0)
        
        # Get embeddings
        hidden_states = self.model.embeddings(input_ids)
        
        B, L, D = hidden_states.shape
        
        # Create mask
        mask = torch.ones(input_ids.shape, device=self.device, dtype=torch.bool)
        
        # Forward through backbone (without lm_head)
        hidden_states, _ = self.model.backbone(
            hidden_states,
            cu_seqlens=None,
            max_seqlen=None,
            mask=mask,
            inference_params=None,
        )
        
        # Reshape back
        hidden_states = hidden_states.view(B, L, D)
        
        return hidden_states
    
    def _pool(self, hidden_states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply pooling to hidden states.
        
        Args:
            hidden_states: Shape (B, L, D)
            mask: Optional boolean mask of shape (B, L)
            
        Returns:
            Pooled features of shape (B, D)
        """
        if self.pooling == 'mean':
            if mask is not None:
                # Masked mean
                mask_expanded = mask.unsqueeze(-1).float()
                summed = (hidden_states * mask_expanded).sum(dim=1)
                lengths = mask_expanded.sum(dim=1).clamp(min=1)
                return summed / lengths
            else:
                return hidden_states.mean(dim=1)
        
        elif self.pooling == 'max':
            if mask is not None:
                # Masked max: set masked positions to large negative
                mask_expanded = mask.unsqueeze(-1)
                hidden_states = hidden_states.masked_fill(~mask_expanded, float('-inf'))
            return hidden_states.max(dim=1).values
        
        elif self.pooling == 'cls':
            # First token (after BOS)
            return hidden_states[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def featurize_single(self, smiles: str) -> np.ndarray:
        """
        Extract features for a single SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Feature vector of shape (D,)
        """
        with torch.no_grad():
            hidden_states = self._extract_hidden_states(smiles)
            pooled = self._pool(hidden_states)
            return pooled.squeeze(0).cpu().float().numpy()
    
    def featurize_batch(self, smiles_list: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Extract features for a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            show_progress: Show progress bar
            
        Returns:
            Feature matrix of shape (N, D)
        """
        features = []
        
        iterator = tqdm(smiles_list, desc=f"Extracting H-Net features ({self.pooling})") if show_progress else smiles_list
        
        for smiles in iterator:
            try:
                feat = self.featurize_single(smiles)
                features.append(feat)
            except Exception as e:
                print(f"Error processing '{smiles}': {e}")
                # Append zeros for failed samples
                features.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (for interpretability)."""
        return [f'hnet_{self.pooling}_{i}' for i in range(self.feature_dim)]
    
    @property
    def name(self) -> str:
        """Get featurizer name for logging."""
        return f"HNet_{self.model_info['run_name']}_{self.pooling}"


def get_all_polymer_featurizers(
    checkpoints_dir: str,
    pooling: PoolingStrategy = 'mean',
    device: str = None,
) -> Dict[str, HNetFeaturizer]:
    """
    Get all polymer H-Net featurizers.
    
    Args:
        checkpoints_dir: Path to checkpoints directory
        pooling: Pooling strategy
        device: Device to run on
        
    Returns:
        Dictionary mapping model name to featurizer
    """
    checkpoints_dir = Path(checkpoints_dir)
    featurizers = {}
    
    for name, run_dir in POLYMER_CHECKPOINTS.items():
        checkpoint_path = checkpoints_dir / run_dir
        if checkpoint_path.exists():
            featurizers[name] = HNetFeaturizer(
                checkpoint_dir=str(checkpoint_path),
                pooling=pooling,
                device=device,
            )
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    return featurizers


def get_all_molecule_featurizers(
    checkpoints_dir: str,
    pooling: PoolingStrategy = 'mean',
    device: str = None,
) -> Dict[str, HNetFeaturizer]:
    """
    Get all molecule H-Net featurizers.
    
    Args:
        checkpoints_dir: Path to checkpoints directory
        pooling: Pooling strategy
        device: Device to run on
        
    Returns:
        Dictionary mapping model name to featurizer
    """
    checkpoints_dir = Path(checkpoints_dir)
    featurizers = {}
    
    for name, run_dir in MOLECULE_CHECKPOINTS.items():
        checkpoint_path = checkpoints_dir / run_dir
        if checkpoint_path.exists():
            featurizers[name] = HNetFeaturizer(
                checkpoint_dir=str(checkpoint_path),
                pooling=pooling,
                device=device,
            )
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    return featurizers




