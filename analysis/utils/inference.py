"""
Utilities for loading H-Net models and running tokenization inference.
"""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.utils.tokenizers import ByteTokenizer
from omegaconf import ListConfig


def get_architecture_type(arch_layout) -> str:
    """
    Determine if architecture is 1-stage or 2-stage from arch_layout.
    
    Args:
        arch_layout: Architecture layout from config (e.g., ["m4", ["T22"], "m4"])
    
    Returns:
        '1-stage' or '2-stage'
    """
    # Convert to string for easier pattern matching
    layout_str = str(arch_layout)
    
    # 2-stage has nested structure like ["m4", ["T1m4", ["T22"], "m4T1"], "m4"]
    # Key indicators: "T1m4" or "m4T1" in the layout
    if 'T1m4' in layout_str or 'm4T1' in layout_str:
        return '2-stage'
    else:
        return '1-stage'


def get_model_info(checkpoint_dir: str) -> Dict:
    """
    Extract model information from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., checkpoints/run_large_20251111_075600)
    
    Returns:
        Dictionary with model metadata
    """
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract key information
    config = metadata.get('config', {})
    training_args = metadata.get('training_args', {})
    dataset_path = training_args.get('data', '')
    
    # Determine architecture type (1-stage vs 2-stage)
    arch_layout = config.get('arch_layout', [])
    architecture_type = get_architecture_type(arch_layout)
    
    info = {
        'run_name': metadata.get('run_name', checkpoint_path.name),
        'dataset': dataset_path,
        'config': config,
        'phase': metadata.get('phase', 'unknown'),
        'checkpoint_path': str(checkpoint_path / 'checkpoints' / 'checkpoint_bytes_best.pt'),
        'metadata': metadata,
        'architecture_type': architecture_type,  # NEW: '1-stage' or '2-stage'
        'arch_layout': arch_layout,
    }
    
    # Determine dataset type and full path
    if 'PI1M' in dataset_path:
        info['dataset_type'] = 'PI1M'
        info['dataset_csv'] = str(project_root / 'datasets' / 'PI1M' / 'PI1M_v2.csv')
    elif 'moses' in dataset_path.lower():
        info['dataset_type'] = 'MOSES'
        info['dataset_csv'] = str(project_root / 'datasets' / 'moses' / 'smiles-molecules-moses_all.csv')
    else:
        info['dataset_type'] = 'unknown'
        info['dataset_csv'] = None
    
    # Check if concatenation was used (from training_args already extracted above)
    info['concatenate'] = training_args.get('concatenate', False)
    info['num_concatenate'] = training_args.get('num_concatenate', 1)
    
    # Extract epochs from metadata (training_history)
    training_history = metadata.get('training_history', [])
    epoch_checkpoints = [h for h in training_history if h.get('checkpoint_type') == 'epoch']
    info['epochs'] = len(epoch_checkpoints) if epoch_checkpoints else training_args.get('epochs', 0)
    
    return info


def load_model(checkpoint_dir: str, device: str = None) -> Tuple[HNetForCausalLM, Dict]:
    """
    Load H-Net model from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        device: Device to load model on ('cuda' or 'cpu'). Auto-detect if None.
    
    Returns:
        Tuple of (model, model_info)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get model info
    info = get_model_info(checkpoint_dir)
    checkpoint_path = info['checkpoint_path']
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Extract config from metadata
    config = info['config']
    
    # Create config objects
    attn_cfg = AttnConfig(**config['attn_cfg'])
    ssm_cfg = SSMConfig(**config['ssm_cfg'])
    
    # Remove nested configs before creating HNetConfig
    config_copy = config.copy()
    config_copy.pop('attn_cfg')
    config_copy.pop('ssm_cfg')
    
    hnet_cfg = HNetConfig(**config_copy, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
    
    # Create model
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=torch.bfloat16)
    model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    major, minor = map(int, torch.__version__.split('.')[:2])
    if (major, minor) >= (2, 6):
        with torch.serialization.safe_globals([ListConfig]):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Training checkpoint format
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded training checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Direct state dict format
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully on {device}")
    
    return model, info


def extract_tokenization(model: HNetForCausalLM, text: str, tokenizer: ByteTokenizer, 
                        device: str) -> Dict[str, any]:
    """
    Run model inference and extract tokenization boundaries.
    
    This follows the same approach as train_smiles.py's save_boundary_predictions()
    and visualize_training_evolution.py's get_boundary_predictions().
    
    Supports both 1-stage and 2-stage H-Net architectures.
    
    Args:
        model: Loaded H-Net model
        text: Input text (SMILES/PSMILES string)
        tokenizer: ByteTokenizer instance
        device: Device model is on
    
    Returns:
        Dictionary with tokenization information:
        - tokens: List of token strings (from Stage 0)
        - breakpoints: List of breakpoint indices (character positions)
        - breakpoint_chars: Characters at breakpoint positions
        - boundary_mask: Binary array indicating boundary positions (Stage 0)
        - boundary_prob: Probability array for boundaries (shape: [L, 2])
        - num_stages: Number of chunking stages (1 or 2)
        - stages: List of stage-specific data (for multi-stage analysis)
    """
    # Tokenize input (add BOS and EOS for model processing)
    encoded = tokenizer.encode([text], add_bos=True, add_eos=True)[0]
    input_ids = torch.tensor(
        encoded["input_ids"], dtype=torch.long, device=device
    ).unsqueeze(0)
    
    # Forward pass to get boundary predictions
    with torch.no_grad():
        mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        output = model.forward(input_ids, mask=mask)
        
        # Extract boundary predictions from ALL stages
        # bpred_outputs is a list of BoundaryPrediction objects, one per stage
        bpred_outputs = output.bpred_output
        
        stages_data = []
        if bpred_outputs and len(bpred_outputs) > 0:
            for stage_idx, bpred in enumerate(bpred_outputs):
                if bpred is not None:
                    stage_boundary_mask = bpred.boundary_mask[0].cpu().numpy()
                    stage_boundary_prob = bpred.boundary_prob[0].cpu().float().numpy()
                    
                    # Remove BOS and EOS tokens for Stage 0
                    if stage_idx == 0:
                        stage_boundary_mask = stage_boundary_mask[1:-1]
                        stage_boundary_prob = stage_boundary_prob[1:-1]
                    # Stage 1+ operate on chunks, no BOS/EOS to remove
                    
                    stages_data.append({
                        'stage': stage_idx,
                        'boundary_mask': stage_boundary_mask.tolist(),
                        'boundary_prob': stage_boundary_prob.tolist(),
                        'num_boundaries': int(stage_boundary_mask.sum()),
                        'length': len(stage_boundary_mask),
                    })
        
        # Fallback if no stages
        if not stages_data:
            fallback_mask = np.zeros(len(encoded['input_ids']) - 2, dtype=bool)
            fallback_mask[0] = True
            stages_data.append({
                'stage': 0,
                'boundary_mask': fallback_mask.tolist(),
                'boundary_prob': np.zeros((len(fallback_mask), 2)).tolist(),
                'num_boundaries': 1,
                'length': len(fallback_mask),
            })
    
    num_stages = len(stages_data)
    
    # Use Stage 0 for primary tokenization (byte-level)
    boundary_mask = np.array(stages_data[0]['boundary_mask'], dtype=bool)
    boundary_prob = np.array(stages_data[0]['boundary_prob'])
    
    # Convert boundary mask to tokens
    # A boundary at position i means a new token starts at position i
    tokens = []
    breakpoints = []
    breakpoint_chars = []
    
    current_token = ""
    for i, char in enumerate(text):
        if i < len(boundary_mask) and boundary_mask[i] and current_token:
            # This is a boundary position - finish current token
            tokens.append(current_token)
            breakpoints.append(i)
            breakpoint_chars.append(char)
            current_token = char
        else:
            # Continue building current token
            current_token += char
    
    # Add the last token
    if current_token:
        tokens.append(current_token)
    
    # For 2-stage models, also compute super-chunks from Stage 1
    super_chunks = []
    stage1_breakpoints = []
    if num_stages > 1:
        stage1_mask = np.array(stages_data[1]['boundary_mask'], dtype=bool)
        # Stage 1 boundaries are over Stage 0 chunks
        # Map back to character positions using token boundaries
        token_start_positions = [0] + breakpoints  # Start of each token
        
        # Group tokens into super-chunks based on Stage 1 boundaries
        current_super_chunk = []
        for chunk_idx in range(len(tokens)):
            if chunk_idx < len(stage1_mask) and stage1_mask[chunk_idx] and current_super_chunk:
                # Stage 1 boundary - finish current super-chunk
                super_chunks.append(''.join(current_super_chunk))
                if chunk_idx < len(token_start_positions):
                    stage1_breakpoints.append(token_start_positions[chunk_idx])
                current_super_chunk = [tokens[chunk_idx]] if chunk_idx < len(tokens) else []
            else:
                if chunk_idx < len(tokens):
                    current_super_chunk.append(tokens[chunk_idx])
        
        if current_super_chunk:
            super_chunks.append(''.join(current_super_chunk))
    
    result = {
        'text': text,
        'tokens': tokens,  # Stage 0 tokens (byte-level)
        'breakpoints': breakpoints,
        'breakpoint_chars': breakpoint_chars,
        'num_tokens': len(tokens),
        'boundary_mask': boundary_mask.tolist(),
        'boundary_prob': boundary_prob.tolist(),
        'num_stages': num_stages,
        'stages': stages_data,  # All stage data for detailed analysis
    }
    
    # Add 2-stage specific data
    if num_stages > 1:
        result['super_chunks'] = super_chunks  # Stage 1 super-chunks
        result['num_super_chunks'] = len(super_chunks)
        result['stage1_breakpoints'] = stage1_breakpoints
        result['chunks_per_super_chunk'] = [len(sc) for sc in super_chunks] if super_chunks else []
    
    return result


def get_stage_statistics(results: List[Dict]) -> Dict:
    """
    Compute aggregate statistics for multi-stage tokenization results.
    
    Args:
        results: List of tokenization results from extract_tokenization()
    
    Returns:
        Dictionary with stage-level statistics
    """
    stats = {
        'total_samples': len(results),
        'stage0': {
            'total_tokens': 0,
            'unique_tokens': set(),
            'token_lengths': [],
            'tokens_per_sample': [],
        }
    }
    
    # Check if we have 2-stage data
    has_stage1 = any(r.get('num_stages', 1) > 1 for r in results)
    if has_stage1:
        stats['stage1'] = {
            'total_super_chunks': 0,
            'unique_super_chunks': set(),
            'super_chunk_lengths': [],
            'super_chunks_per_sample': [],
            'chunks_per_super_chunk': [],
        }
    
    for result in results:
        # Stage 0 stats
        tokens = result.get('tokens', [])
        stats['stage0']['total_tokens'] += len(tokens)
        stats['stage0']['unique_tokens'].update(tokens)
        stats['stage0']['token_lengths'].extend([len(t) for t in tokens])
        stats['stage0']['tokens_per_sample'].append(len(tokens))
        
        # Stage 1 stats (if present)
        if has_stage1 and result.get('num_stages', 1) > 1:
            super_chunks = result.get('super_chunks', [])
            stats['stage1']['total_super_chunks'] += len(super_chunks)
            stats['stage1']['unique_super_chunks'].update(super_chunks)
            stats['stage1']['super_chunk_lengths'].extend([len(sc) for sc in super_chunks])
            stats['stage1']['super_chunks_per_sample'].append(len(super_chunks))
            stats['stage1']['chunks_per_super_chunk'].extend(
                result.get('chunks_per_super_chunk', [])
            )
    
    # Convert sets to counts for JSON serialization
    stats['stage0']['num_unique_tokens'] = len(stats['stage0']['unique_tokens'])
    stats['stage0']['unique_tokens'] = list(stats['stage0']['unique_tokens'])
    
    if has_stage1:
        stats['stage1']['num_unique_super_chunks'] = len(stats['stage1']['unique_super_chunks'])
        stats['stage1']['unique_super_chunks'] = list(stats['stage1']['unique_super_chunks'])
    
    return stats


def run_tokenization_inference(model: HNetForCausalLM, dataset_csv: str, 
                               dataset_type: str, device: str,
                               max_samples: Optional[int] = None,
                               batch_size: int = 32) -> List[Dict]:
    """
    Run tokenization inference on entire dataset.
    
    Args:
        model: Loaded H-Net model
        dataset_csv: Path to dataset CSV file
        dataset_type: 'PI1M' or 'MOSES'
        device: Device model is on
        max_samples: Maximum number of samples to process (None = all)
        batch_size: Batch size for processing (for efficiency)
    
    Returns:
        List of tokenization results, one per SMILES string
    """
    tokenizer = ByteTokenizer()
    
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
    
    print(f"Processing {len(smiles_list)} SMILES strings...")
    
    results = []
    for smiles in tqdm(smiles_list, desc="Tokenizing"):
        try:
            result = extract_tokenization(model, smiles, tokenizer, device)
            results.append(result)
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            # Store error result
            results.append({
                'text': smiles,
                'tokens': [],
                'breakpoints': [],
                'breakpoint_chars': [],
                'num_tokens': 0,
                'error': str(e)
            })
    
    return results


def save_tokenization_results(results: List[Dict], output_path: str):
    """
    Save tokenization results to file.
    
    Args:
        results: List of tokenization results
        output_path: Path to save results (supports .json, .pkl, .parquet)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.json':
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif output_path.suffix == '.pkl':
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    elif output_path.suffix == '.parquet':
        # Convert to DataFrame
        df = pd.DataFrame(results)
        df.to_parquet(output_path, compression='gzip')
    else:
        raise ValueError(f"Unsupported file format: {output_path.suffix}")
    
    print(f"Results saved to {output_path}")


def load_tokenization_results(input_path: str) -> List[Dict]:
    """
    Load tokenization results from file.
    
    Args:
        input_path: Path to results file
    
    Returns:
        List of tokenization results
    """
    input_path = Path(input_path)
    
    if input_path.suffix == '.json':
        import json
        with open(input_path, 'r') as f:
            return json.load(f)
    elif input_path.suffix == '.pkl':
        import pickle
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    elif input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

