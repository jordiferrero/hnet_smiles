#!/usr/bin/env python3
"""
Visualize how chunk boundaries evolve as the model trains on more data.
Shows the same SMILES string across different training checkpoints (epochs),
displaying how boundaries change as training bytes increase.
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import gzip

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.utils.tokenizers import ByteTokenizer
from visualizations.utils import (
    draw_chunking_visualization,
    draw_multistage_chunking_visualization,
    get_chunk_spans,
)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model_from_checkpoint(checkpoint_path: str, config_dict: dict, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"[DEBUG]   Creating model config...")
    # Make a copy to avoid modifying the original
    config_copy = config_dict.copy()
    
    # Remove training_config if present (not part of model config)
    config_copy.pop("training_config", None)
    
    attn_cfg = AttnConfig(**config_copy.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config_copy.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config_copy, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
    
    print(f"[DEBUG]   Instantiating model on {device}...")
    dtype = torch.bfloat16 if device.type != 'cpu' else torch.float32
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
    model.eval()
    
    print(f"[DEBUG]   Loading checkpoint weights from {Path(checkpoint_path).name}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        print(f"[DEBUG]   Loading model_state_dict...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"[DEBUG]   Loading checkpoint directly...")
        model.load_state_dict(checkpoint)
    
    print(f"[DEBUG]   Model loaded and ready")
    return model, checkpoint


def get_boundary_predictions(
    model: HNetForCausalLM,
    tokenizer: ByteTokenizer,
    text: str,
    device: torch.device,
) -> tuple:
    """
    Get boundary predictions for a text sequence.
    Supports both 1-stage and 2-stage H-Net models.
    
    Returns:
        (boundary_mask, boundary_prob, text, stages_data)
        - boundary_mask: Stage 0 boundary mask (for backward compat)
        - boundary_prob: Stage 0 boundary prob (for backward compat)
        - text: Input text
        - stages_data: List of dicts with 'boundary_mask' and 'boundary_prob' for each stage
    """
    # Tokenize
    encoded = tokenizer.encode([text], add_bos=True, add_eos=True)[0]
    input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        output = model.forward(input_ids, mask=mask)
        
        # Extract boundary predictions from ALL stages
        bpred_outputs = output.bpred_output
        stages_data = []
        
        if bpred_outputs and len(bpred_outputs) > 0:
            for stage_idx, bpred in enumerate(bpred_outputs):
                if bpred is not None:
                    stage_mask = bpred.boundary_mask[0].cpu().numpy()  # (L,)
                    stage_prob = bpred.boundary_prob[0].cpu().float().numpy()  # (L, 2)
                    
                    # Only remove BOS/EOS tokens for Stage 0 (character level)
                    # Stage 1+ operates on chunks, not characters - no BOS/EOS to remove
                    if stage_idx == 0:
                        stage_mask = stage_mask[1:-1]
                        stage_prob = stage_prob[1:-1]
                    
                    stages_data.append({
                        'stage': stage_idx,
                        'boundary_mask': stage_mask,
                        'boundary_prob': stage_prob,
                    })
        
        # Fallback if no stages
        if not stages_data:
            fallback_mask = np.zeros(len(encoded['input_ids']) - 2, dtype=bool)
            fallback_mask[0] = True
            fallback_prob = np.zeros((len(encoded['input_ids']) - 2, 2))
            stages_data.append({
                'stage': 0,
                'boundary_mask': fallback_mask,
                'boundary_prob': fallback_prob,
            })
    
    # Return stage 0 for backward compat, plus all stages
    boundary_mask = stages_data[0]['boundary_mask']
    boundary_prob = stages_data[0]['boundary_prob']
    
    return boundary_mask, boundary_prob, text, stages_data


def create_multi_row_frame(
    texts_data: list,
    epoch: int,
    training_bytes: int,
    figsize: tuple = (16, 12),
    dpi: int = 100,
    is_concatenated: bool = False,
    chars_per_row: int = 80,
    num_stages: int = 1,
) -> plt.Figure:
    """
    Create a single frame showing chunking for multiple SMILES strings in rows.
    Only shows the chunking visualization column.
    Supports both 1-stage and 2-stage visualization.
    
    Args:
        texts_data: List of dicts, each with 'text', 'boundary_mask', 'boundary_prob', and optionally 'stages'
        epoch: Current epoch number (for reference)
        training_bytes: Training bytes seen so far
        figsize: Figure size
        dpi: DPI for figure
        is_concatenated: If True, show only first sample wrapped across multiple rows
        chars_per_row: Number of characters per row when wrapping concatenated samples
        num_stages: Number of chunking stages (1 or 2)
    """
    # Store original stages data for concatenated mode
    original_stages_data = None
    
    if is_concatenated and texts_data:
        # For concatenated samples, wrap the first sample across multiple rows
        first_data = texts_data[0]
        text = first_data['text']
        boundary_mask = first_data['boundary_mask']
        boundary_prob = first_data['boundary_prob']
        original_stages_data = first_data.get('stages', None)  # Preserve stages data
        
        # Split the text and boundaries into chunks for each row
        text_len = len(text)
        num_rows = (text_len + chars_per_row - 1) // chars_per_row  # Ceiling division
        
        # Create row data for wrapped display
        wrapped_rows = []
        for row_idx in range(num_rows):
            start_idx = row_idx * chars_per_row
            end_idx = min(start_idx + chars_per_row, text_len)
            
            row_text = text[start_idx:end_idx]
            row_boundary_mask = boundary_mask[start_idx:end_idx]
            row_boundary_prob = boundary_prob[start_idx:end_idx]
            
            wrapped_rows.append({
                'text': row_text,
                'boundary_mask': row_boundary_mask,
                'boundary_prob': row_boundary_prob,
                'start_char': start_idx,  # For x-axis offset
                'stages': original_stages_data,  # Pass full stages data for reference
                'full_text': text,  # Pass full text for multi-stage rendering
                'full_boundary_mask': boundary_mask,  # Pass full mask
            })
        
        texts_data = wrapped_rows
        num_rows = len(wrapped_rows)
    else:
        num_rows = len(texts_data)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Create a grid: 1 column per row (only chunking visualization)
    gs = fig.add_gridspec(num_rows, 1, hspace=0.2, wspace=0.1, 
                          left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Find max text length for consistent x-axis
    if is_concatenated and texts_data:
        # For wrapped rows, use chars_per_row as max length
        max_text_len = chars_per_row
    else:
        max_text_len = max(len(data['text']) for data in texts_data) if texts_data else 50
    
    for row_idx, data in enumerate(texts_data):
        text = data['text']
        boundary_mask = data['boundary_mask']
        boundary_prob = data['boundary_prob']
        start_char = data.get('start_char', 0)  # For wrapped rows, this is the offset
        
        # Chunking visualization only
        ax_chunk = fig.add_subplot(gs[row_idx, 0])
        
        if is_concatenated:
            # For wrapped rows, set x-axis to show the full character range for this row
            # Use chars_per_row to ensure all rows have the same box size
            ax_chunk.set_xlim(start_char, start_char + chars_per_row)
        else:
            ax_chunk.set_xlim(0, max_text_len)
        
        ax_chunk.set_ylim(-0.5, 1.5)
        ax_chunk.set_aspect('equal')
        ax_chunk.axis('off')
        
        # Draw chunking visualization
        # For wrapped rows, we need to adjust the x-coordinates
        if is_concatenated:
            # Draw with offset by manually adjusting coordinates
            # This supports both 1-stage and 2-stage visualization
            num_chars = len(text)
            square_size = 0.8
            
            # Check if we have multi-stage data
            stages_data = data.get('stages', None)
            has_multistage = stages_data and len(stages_data) > 1
            
            if has_multistage:
                # Multi-stage layout: Stage0 boundary (2.5), Stage0 spans (1.5), Stage1 spans (0.5), chars (-0.5)
                ax_chunk.set_ylim(-1.5, 4.0)
                y_stage0_boundary = 2.5
                y_stage0_spans = 1.5
                y_stage1_spans = 0.5
                y_chars = -0.5
                
                # Get full data for multi-stage calculations
                full_text = data.get('full_text', text)
                full_boundary_mask = data.get('full_boundary_mask', boundary_mask)
                
                # Stage 0: Draw boundary markers for this row's portion
                for i in range(num_chars):
                    color = '#90EE90' if boundary_mask[i] else '#FFFFFF'
                    square = mpatches.Rectangle(
                        (start_char + i, y_stage0_boundary - square_size/2),
                        square_size, square_size,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    ax_chunk.add_patch(square)
                
                # Stage 0: Draw chunk spans for this row
                row_spans = get_chunk_spans(boundary_mask)
                for start, end in row_spans:
                    if end > num_chars:
                        end = num_chars
                    width = end - start
                    if width > 0:
                        ax_chunk.plot([start_char + start, start_char + start], 
                                     [y_stage0_spans, y_stage0_spans + square_size], 
                                     'b-', linewidth=2, alpha=0.7)
                        ax_chunk.plot([start_char + start, start_char + end], 
                                     [y_stage0_spans, y_stage0_spans], 
                                     'b-', linewidth=2, alpha=0.7)
                        ax_chunk.plot([start_char + end, start_char + end], 
                                     [y_stage0_spans, y_stage0_spans + square_size], 
                                     'b-', linewidth=2, alpha=0.7)
                        rect = mpatches.Rectangle(
                            (start_char + start, y_stage0_spans), width, square_size,
                            facecolor='#ADD8E6', alpha=0.3, edgecolor='none'
                        )
                        ax_chunk.add_patch(rect)
                
                # Stage 1: Draw super-chunk spans that overlap with this row
                if len(stages_data) > 1:
                    stage1_mask = stages_data[1]['boundary_mask']
                    # Get full Stage 0 spans to map Stage 1 boundaries
                    full_stage0_spans = get_chunk_spans(full_boundary_mask)
                    stage1_spans = get_chunk_spans(stage1_mask)
                    
                    for chunk_start, chunk_end in stage1_spans:
                        if chunk_start < len(full_stage0_spans) and chunk_end <= len(full_stage0_spans):
                            # Map to character positions
                            char_start = full_stage0_spans[chunk_start][0]
                            char_end = full_stage0_spans[min(chunk_end, len(full_stage0_spans)) - 1][1] if chunk_end > 0 else char_start
                            
                            # Check if this super-chunk overlaps with current row
                            row_end = start_char + num_chars
                            if char_end > start_char and char_start < row_end:
                                # Clip to row boundaries
                                draw_start = max(char_start, start_char)
                                draw_end = min(char_end, row_end)
                                width = draw_end - draw_start
                                if width > 0:
                                    ax_chunk.plot([draw_start, draw_start], 
                                                 [y_stage1_spans, y_stage1_spans + square_size], 
                                                 'purple', linewidth=3, alpha=0.8)
                                    ax_chunk.plot([draw_start, draw_end], 
                                                 [y_stage1_spans, y_stage1_spans], 
                                                 'purple', linewidth=3, alpha=0.8)
                                    ax_chunk.plot([draw_end, draw_end], 
                                                 [y_stage1_spans, y_stage1_spans + square_size], 
                                                 'purple', linewidth=3, alpha=0.8)
                                    rect = mpatches.Rectangle(
                                        (draw_start, y_stage1_spans), width, square_size,
                                        facecolor='#E6E6FA', alpha=0.4, edgecolor='none'
                                    )
                                    ax_chunk.add_patch(rect)
                
                # Draw character row
                for i in range(num_chars):
                    square = mpatches.Rectangle(
                        (start_char + i, y_chars),
                        square_size, square_size,
                        facecolor='#FFFFFF',
                        edgecolor='black',
                        linewidth=0.5
                    )
                    ax_chunk.add_patch(square)
                    ax_chunk.text(start_char + i + 0.4, y_chars + square_size/2, text[i],
                                 ha='center', va='center', fontsize=8)
            else:
                # Single-stage layout (original code)
                y_top = 1.0
                y_bottom = 0.0
                
                # Draw top row: boundary markers (green squares)
                for i in range(num_chars):
                    color = '#90EE90' if boundary_mask[i] else '#FFFFFF'
                    square = mpatches.Rectangle(
                        (start_char + i, y_top - square_size/2),
                        square_size, square_size,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    ax_chunk.add_patch(square)
                
                # Draw bottom row: chunk spans (blue U-shapes)
                spans = get_chunk_spans(boundary_mask)
                
                for start, end in spans:
                    if end > num_chars:
                        end = num_chars
                    
                    width = end - start
                    if width > 0:
                        # Left vertical line
                        ax_chunk.plot([start_char + start, start_char + start], 
                                     [y_bottom, y_bottom + square_size], 
                                     'b-', linewidth=2, alpha=0.7)
                        # Bottom horizontal line
                        ax_chunk.plot([start_char + start, start_char + end], 
                                     [y_bottom, y_bottom], 
                                     'b-', linewidth=2, alpha=0.7)
                        # Right vertical line
                        ax_chunk.plot([start_char + end, start_char + end], 
                                     [y_bottom, y_bottom + square_size], 
                                     'b-', linewidth=2, alpha=0.7)
                        
                        # Fill with light blue
                        rect = mpatches.Rectangle(
                            (start_char + start, y_bottom),
                            width, square_size,
                            facecolor='#ADD8E6',
                            alpha=0.3,
                            edgecolor='none'
                        )
                        ax_chunk.add_patch(rect)
                
                # Draw white squares in bottom row
                for i in range(num_chars):
                    square = mpatches.Rectangle(
                        (start_char + i, y_bottom),
                        square_size, square_size,
                        facecolor='#FFFFFF',
                        edgecolor='black',
                        linewidth=0.5
                    )
                    ax_chunk.add_patch(square)
                
                # Add character positions
                for i in range(num_chars):
                    ax_chunk.text(start_char + i + 0.4, y_bottom + square_size/2, text[i],
                                 ha='center', va='center', fontsize=8)
        else:
            # Check if we have multi-stage data
            stages_data = data.get('stages', None)
            if stages_data and len(stages_data) > 1:
                # Multi-stage visualization
                # Layout: Stage0 boundary (2.5), Stage0 spans (1.5), Stage1 spans (0.5), chars (-0.5)
                ax_chunk.set_ylim(-1.5, 4.0)
                ax_chunk.set_xlim(-8, max_text_len + 1)  # Extra space for labels
                draw_multistage_chunking_visualization(
                    ax_chunk, text, stages_data, current_length=len(text)
                )
            else:
                # Single-stage visualization (backward compat)
                draw_chunking_visualization(
                    ax_chunk, text, boundary_mask, boundary_prob, current_length=len(text)
                )
    
    # Add training info at the top - format bytes nicely
    if training_bytes >= 1_000_000_000:
        training_bytes_str = f"{training_bytes/1_000_000_000:.2f}B"
    elif training_bytes >= 1_000_000:
        training_bytes_str = f"{training_bytes/1_000_000:.2f}M"
    elif training_bytes >= 1_000:
        training_bytes_str = f"{training_bytes/1_000:.2f}K"
    else:
        training_bytes_str = f"{training_bytes:,}"
    
    # Include stage info in title
    stage_str = f"{num_stages} stage{'s' if num_stages > 1 else ''}"
    
    fig.suptitle(
        f'Training Bytes: {training_bytes_str} | Epoch: {epoch} | {stage_str}',
        fontsize=14, y=0.98, fontweight='bold'
    )
    
    return fig


def load_smiles_from_dataset(metadata_path: Path, num_samples: int = 10) -> list:
    """Load SMILES strings from the training dataset or test_smiles.txt if available."""
    # First, try to load from test_smiles.txt (if it exists)
    run_dir = metadata_path.parent
    test_smiles_file = run_dir / "test_smiles.txt"
    
    if test_smiles_file.exists():
        print(f"Loading SMILES from test_smiles.txt...")
        with open(test_smiles_file, 'r') as f:
            smiles = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(smiles)} SMILES from test_smiles.txt")
        return smiles[:num_samples]
    
    # Fallback: load from training dataset
    import pandas as pd
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get dataset path from metadata
    data_path = metadata['training_args']['data']
    max_samples = metadata['training_args'].get('max_samples')
    
    # Load dataset
    df = pd.read_csv(data_path)
    smiles_col = df.columns[0]
    smiles = df[smiles_col].astype(str).tolist()
    
    # Apply same filtering as training (if max_samples was used)
    if max_samples:
        smiles = smiles[:max_samples]
    
    # Return top n samples
    return smiles[:num_samples]


def visualize_training_evolution(
    run_dir: Path,
    num_samples: int = 10,
    output_path: str = None,
    checkpoint_pattern: str = "checkpoint_bytes_*.pt",
    max_checkpoints: int = None,
):
    """
    Create animated GIF showing how chunk boundaries evolve across training bytes.
    First tries to load saved visualization images, otherwise generates from checkpoints.
    
    Args:
        run_dir: Directory containing the training run (with checkpoints and metadata)
        num_samples: Number of SMILES strings to visualize (default: 10)
        output_path: Path to save the output GIF (default: run_dir/visualizations/training_evolution.gif)
        checkpoint_pattern: Pattern to match checkpoint files (default: checkpoint_bytes_*.pt)
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Load metadata
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    # Load num_visualize from metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    num_visualize = metadata.get('training_args', {}).get('num_visualize', num_samples)
    num_test_samples = metadata.get('training_args', {}).get('num_test_samples', num_samples)
    is_concatenated = metadata.get('training_args', {}).get('concatenate', False)
    
    print(f"Using num_visualize={num_visualize} samples per image (from config)")
    print(f"Total test samples: {num_test_samples}")
    print(f"Concatenated samples: {is_concatenated}")
    
    # First, try to load saved boundary predictions from pickle files (new format)
    predictions_dir = run_dir / "visualizations" / "predictions"
    if predictions_dir.exists():
        # Find all prediction files
        prediction_files = sorted(predictions_dir.glob("predictions_bytes_*.pkl.gz"))
        
        if prediction_files:
            print(f"Loading boundary predictions from {len(prediction_files)} pickle files...")
            
            # Sort prediction files by bytes threshold (extract from filename)
            def get_bytes_from_filename(pred_file):
                # Filename format: predictions_bytes_1,000.pkl.gz
                # .stem gives: predictions_bytes_1,000.pkl
                name = pred_file.stem  # Remove .gz extension
                if name.endswith('.pkl'):
                    name = name[:-4]  # Remove .pkl
                if 'bytes_' in name:
                    bytes_str = name.split('bytes_')[1].replace(',', '')
                    try:
                        return int(bytes_str)
                    except:
                        return 0
                return 0
            
            prediction_files = sorted(prediction_files, key=get_bytes_from_filename)
            
            # Load all prediction files
            entries = []
            for pred_file in tqdm(prediction_files, desc="Loading prediction files"):
                try:
                    with gzip.open(pred_file, 'rb') as f:
                        data_entry = pickle.load(f)
                    entries.append(data_entry)
                except Exception as e:
                    print(f"Warning: Failed to load {pred_file}: {e}")
                    continue
            
            if entries:
                print(f"Found {len(entries)} prediction entries")
                
                # Sort entries by training bytes
                entries.sort(key=lambda x: x.get('training_bytes', 0))
                
                # Limit number of entries if requested (evenly spaced)
                if max_checkpoints and len(entries) > max_checkpoints:
                    print(f"Limiting to {max_checkpoints} evenly spaced entries...")
                    indices = np.linspace(0, len(entries) - 1, max_checkpoints, dtype=int)
                    entries = [entries[i] for i in indices]
                
                # Get all test samples from first entry (they should be the same across entries)
                first_entry_predictions = entries[0].get('predictions', [])
                total_test_samples = len(first_entry_predictions)
                
                # Calculate number of GIF batches needed
                # For concatenated samples, create one GIF per sample (each wrapped across multiple rows)
                if is_concatenated:
                    num_batches = total_test_samples
                    print(f"Concatenated mode: Creating {num_batches} GIFs (one per sample, each wrapped across multiple rows)")
                else:
                    num_batches = (total_test_samples + num_visualize - 1) // num_visualize  # Ceiling division
                    print(f"Creating {num_batches} separate GIFs with up to {num_visualize} samples each")
                
                # Determine output directory and base filename
                output_dir = Path(output_path).parent
                output_base = Path(output_path).stem  # filename without extension
                
                # Create one GIF per batch of SMILES
                for batch_idx in range(num_batches):
                    if is_concatenated:
                        # For concatenated, one sample per GIF
                        start_idx = batch_idx
                        end_idx = batch_idx + 1
                        print(f"\nGenerating GIF {batch_idx + 1}/{num_batches} for concatenated sample {batch_idx} (wrapped across multiple rows)...")
                    else:
                        start_idx = batch_idx * num_visualize
                        end_idx = min(start_idx + num_visualize, total_test_samples)
                        print(f"\nGenerating GIF {batch_idx + 1}/{num_batches} for SMILES {start_idx}-{end_idx-1}...")
                    
                    # Generate frames for this batch across all training steps
                    batch_frames = []
                    
                    for entry in tqdm(entries, desc=f"Processing batch {batch_idx + 1}"):
                        epoch = entry.get('epoch', 0)
                        training_bytes = entry.get('training_bytes', 0)
                        predictions = entry.get('predictions', [])
                        
                        # Get predictions for this batch
                        batch_predictions = predictions[start_idx:end_idx]
                        
                        # For concatenated samples, each GIF shows one sample wrapped across rows
                        # (start_idx and end_idx already handle this, but keep for clarity)
                        if is_concatenated:
                            batch_predictions = batch_predictions[:1]
                        
                        # Convert batch predictions to texts_data format
                        texts_data = []
                        entry_num_stages = entry.get('num_stages', 1)
                        
                        for pred in batch_predictions:
                            smiles = pred['smiles']
                            # Handle both numpy arrays (new format) and lists (old format)
                            boundary_mask = pred['boundary_mask']
                            if not isinstance(boundary_mask, np.ndarray):
                                boundary_mask = np.array(boundary_mask, dtype=bool)
                            else:
                                boundary_mask = boundary_mask.astype(bool)
                            
                            boundary_prob = pred['boundary_prob']
                            if not isinstance(boundary_prob, np.ndarray):
                                boundary_prob = np.array(boundary_prob, dtype=np.float32)
                            else:
                                boundary_prob = boundary_prob.astype(np.float32)
                            
                            # Get stages data if available (for 2-stage models)
                            stages_data = pred.get('stages', None)
                            if stages_data:
                                # Convert stages data arrays
                                for stage in stages_data:
                                    if not isinstance(stage['boundary_mask'], np.ndarray):
                                        stage['boundary_mask'] = np.array(stage['boundary_mask'], dtype=bool)
                                    else:
                                        stage['boundary_mask'] = stage['boundary_mask'].astype(bool)
                                    if not isinstance(stage['boundary_prob'], np.ndarray):
                                        stage['boundary_prob'] = np.array(stage['boundary_prob'], dtype=np.float32)
                                    else:
                                        stage['boundary_prob'] = stage['boundary_prob'].astype(np.float32)
                            
                            texts_data.append({
                                'text': smiles,
                                'boundary_mask': boundary_mask,
                                'boundary_prob': boundary_prob,
                                'stages': stages_data,  # Include multi-stage data
                            })
                        
                        # Create visualization frame for this batch at this training step
                        fig = create_multi_row_frame(
                            texts_data=texts_data,
                            epoch=epoch,
                            training_bytes=training_bytes,
                            is_concatenated=is_concatenated,
                            num_stages=entry_num_stages,
                        )
                        
                        # Convert to image
                        fig.canvas.draw()
                        buf = fig.canvas.buffer_rgba()
                        img = np.asarray(buf)
                        batch_frames.append(img)
                        
                        plt.close(fig)
                    
                    # Add final frame a few times for pause
                    print("Adding pause frames...")
                    for _ in range(5):
                        if batch_frames:
                            batch_frames.append(batch_frames[-1])
                    
                    # Save this batch as a separate GIF
                    batch_output_path = output_dir / f"{output_base}_batch_{batch_idx + 1}.gif"
                    print(f"\nSaving GIF {batch_idx + 1}/{num_batches} to {batch_output_path}...")
                    imageio.mimsave(batch_output_path, batch_frames, duration=0.5, loop=0)
                    print(f"Saved {len(batch_frames)} frames to {batch_output_path}")
                
                print(f"\n✓ Generated {num_batches} GIFs successfully!")
                return
            else:
                print("No entries loaded from prediction files, falling back to checkpoint-based generation...")
        else:
            print("No prediction files found, trying old JSON format...")
    
    # Fallback: Try old JSON format (for backward compatibility)
    predictions_file = run_dir / "visualizations" / "boundary_predictions.json"
    if predictions_file.exists():
        print(f"Loading boundary predictions from old JSON format: {predictions_file}...")
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        entries = predictions_data.get('entries', [])
        if entries:
            print(f"Found {len(entries)} prediction entries in JSON format")
            
            # Sort entries by training bytes
            entries.sort(key=lambda x: x.get('training_bytes', 0))
            
            # Limit number of entries if requested (evenly spaced)
            if max_checkpoints and len(entries) > max_checkpoints:
                print(f"Limiting to {max_checkpoints} evenly spaced entries...")
                indices = np.linspace(0, len(entries) - 1, max_checkpoints, dtype=int)
                entries = [entries[i] for i in indices]
            
            # Get all test samples from first entry (they should be the same across entries)
            first_entry_predictions = entries[0].get('predictions', [])
            total_test_samples = len(first_entry_predictions)
            
            # Calculate number of GIF batches needed
            # For concatenated samples, create one GIF per sample (each wrapped across multiple rows)
            if is_concatenated:
                num_batches = total_test_samples
                print(f"Concatenated mode: Creating {num_batches} GIFs (one per sample, each wrapped across multiple rows)")
            else:
                num_batches = (total_test_samples + num_visualize - 1) // num_visualize  # Ceiling division
                print(f"Creating {num_batches} separate GIFs with up to {num_visualize} samples each")
            
            # Determine output directory and base filename
            output_dir = Path(output_path).parent
            output_base = Path(output_path).stem  # filename without extension
            
            # Create one GIF per batch of SMILES
            for batch_idx in range(num_batches):
                if is_concatenated:
                    # For concatenated, one sample per GIF
                    start_idx = batch_idx
                    end_idx = batch_idx + 1
                    print(f"\nGenerating GIF {batch_idx + 1}/{num_batches} for concatenated sample {batch_idx} (wrapped across multiple rows)...")
                else:
                    start_idx = batch_idx * num_visualize
                    end_idx = min(start_idx + num_visualize, total_test_samples)
                    print(f"\nGenerating GIF {batch_idx + 1}/{num_batches} for SMILES {start_idx}-{end_idx-1}...")
                
                # Generate frames for this batch across all training steps
                batch_frames = []
                
                for entry in tqdm(entries, desc=f"Processing batch {batch_idx + 1}"):
                    epoch = entry.get('epoch', 0)
                    training_bytes = entry.get('training_bytes', 0)
                    predictions = entry.get('predictions', [])
                    
                    # Get predictions for this batch
                    batch_predictions = predictions[start_idx:end_idx]
                    
                    # For concatenated samples, each GIF shows one sample wrapped across rows
                    # (start_idx and end_idx already handle this, but keep for clarity)
                    if is_concatenated:
                        batch_predictions = batch_predictions[:1]
                    
                    # Convert batch predictions to texts_data format
                    texts_data = []
                    entry_num_stages = entry.get('num_stages', 1)
                    
                    for pred in batch_predictions:
                        smiles = pred['smiles']
                        # Handle both numpy arrays (new format) and lists (old format)
                        boundary_mask = pred['boundary_mask']
                        if not isinstance(boundary_mask, np.ndarray):
                            boundary_mask = np.array(boundary_mask, dtype=bool)
                        else:
                            boundary_mask = boundary_mask.astype(bool)
                        
                        boundary_prob = pred['boundary_prob']
                        if not isinstance(boundary_prob, np.ndarray):
                            boundary_prob = np.array(boundary_prob, dtype=np.float32)
                        else:
                            boundary_prob = boundary_prob.astype(np.float32)
                        
                        # Get stages data if available (for 2-stage models)
                        stages_data = pred.get('stages', None)
                        if stages_data:
                            for stage in stages_data:
                                if not isinstance(stage['boundary_mask'], np.ndarray):
                                    stage['boundary_mask'] = np.array(stage['boundary_mask'], dtype=bool)
                                else:
                                    stage['boundary_mask'] = stage['boundary_mask'].astype(bool)
                                if not isinstance(stage['boundary_prob'], np.ndarray):
                                    stage['boundary_prob'] = np.array(stage['boundary_prob'], dtype=np.float32)
                                else:
                                    stage['boundary_prob'] = stage['boundary_prob'].astype(np.float32)
                        
                        texts_data.append({
                            'text': smiles,
                            'boundary_mask': boundary_mask,
                            'boundary_prob': boundary_prob,
                            'stages': stages_data,
                        })
                    
                    # Create visualization frame for this batch at this training step
                    fig = create_multi_row_frame(
                        texts_data=texts_data,
                        epoch=epoch,
                        training_bytes=training_bytes,
                        is_concatenated=is_concatenated,
                        num_stages=entry_num_stages,
                    )
                    
                    # Convert to image
                    fig.canvas.draw()
                    buf = fig.canvas.buffer_rgba()
                    img = np.asarray(buf)
                    batch_frames.append(img)
                    
                    plt.close(fig)
                
                # Add final frame a few times for pause
                print("Adding pause frames...")
                for _ in range(5):
                    if batch_frames:
                        batch_frames.append(batch_frames[-1])
                
                # Save this batch as a separate GIF
                batch_output_path = output_dir / f"{output_base}_batch_{batch_idx + 1}.gif"
                print(f"\nSaving GIF {batch_idx + 1}/{num_batches} to {batch_output_path}...")
                imageio.mimsave(batch_output_path, batch_frames, duration=0.5, loop=0)
                print(f"Saved {len(batch_frames)} frames to {batch_output_path}")
            
            print(f"\n✓ Generated {num_batches} GIFs successfully!")
            return
        else:
            print("No entries found in JSON file, falling back to checkpoint-based generation...")
    else:
        print("No predictions found (neither pickle nor JSON), falling back to checkpoint-based generation...")
    
    # Fallback: generate from checkpoints (old behavior)
    print("No saved images found. Generating from checkpoints...")
    
    # Load SMILES from training dataset
    # For concatenated samples, load all samples (will create one GIF per sample)
    samples_to_load = num_test_samples if is_concatenated else num_samples
    print(f"Loading {samples_to_load} SMILES from training dataset...")
    texts = load_smiles_from_dataset(metadata_path, samples_to_load)
    print(f"Loaded {len(texts)} SMILES strings")
    
    # Find all checkpoints - try byte-based first, then epoch-based as fallback
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_files = sorted(checkpoints_dir.glob(checkpoint_pattern))
    
    # If no byte-based checkpoints found, try epoch-based
    if not checkpoint_files:
        checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
    
    # If still no checkpoints, try all checkpoints
    if not checkpoint_files:
        checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_*.pt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    
    print(f"Found {len(checkpoint_files)} checkpoints (before sorting)")
    
    # Sort checkpoints by extracting bytes from filename (much faster than loading checkpoints)
    # Filename format: checkpoint_bytes_1,000,000.pt or checkpoint_epoch_1.pt
    def get_bytes_from_filename(checkpoint_path):
        name = checkpoint_path.stem  # Remove .pt extension
        if 'bytes_' in name:
            # Extract bytes from filename (e.g., "checkpoint_bytes_1,000,000" -> 1000000)
            bytes_str = name.split('bytes_')[1].replace(',', '')
            try:
                return int(bytes_str)
            except:
                return 0
        elif 'epoch_' in name:
            # For epoch checkpoints, return a very large number to sort them last
            return 999_999_999
        else:
            return 0
    
    print("[DEBUG] Sorting checkpoints by bytes (from filenames)...")
    checkpoint_files = sorted(checkpoint_files, key=get_bytes_from_filename)
    
    print(f"Sorted {len(checkpoint_files)} checkpoints")
    print(f"Checkpoint files (first 5): {[f.name for f in checkpoint_files[:5]]}")
    print(f"Checkpoint files (last 2): {[f.name for f in checkpoint_files[-2:]]}")
    
    # Limit number of checkpoints if requested (evenly spaced)
    if max_checkpoints and len(checkpoint_files) > max_checkpoints:
        print(f"[DEBUG] Limiting to {max_checkpoints} evenly spaced checkpoints...")
        indices = np.linspace(0, len(checkpoint_files) - 1, max_checkpoints, dtype=int)
        checkpoint_files = [checkpoint_files[i] for i in indices]
        print(f"[DEBUG] Selected checkpoints: {[f.name for f in checkpoint_files]}")
    
    # Load config from first checkpoint
    print(f"\n[DEBUG] Loading config from first checkpoint: {checkpoint_files[0].name}")
    first_checkpoint = torch.load(checkpoint_files[0], map_location='cpu')
    config_dict = first_checkpoint.get('config')
    if not config_dict:
        print("[DEBUG] Config not in checkpoint, loading from metadata...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        config_dict = metadata['config']
    print("[DEBUG] Config loaded successfully")
    
    # Load tokenizer
    print("[DEBUG] Initializing tokenizer...")
    tokenizer = ByteTokenizer()
    print("[DEBUG] Tokenizer initialized")
    
    # Process checkpoints
    # For concatenated samples, create one GIF per sample
    # For non-concatenated, create one GIF with all samples
    
    if is_concatenated:
        # Create one GIF per sample
        output_dir = Path(output_path).parent
        output_base = Path(output_path).stem  # filename without extension
        
        print(f"\n[DEBUG] Concatenated mode: Creating {len(texts)} separate GIFs (one per sample)")
        
        for sample_idx, sample_text in enumerate(texts):
            print(f"\n[DEBUG] Processing sample {sample_idx + 1}/{len(texts)}...")
            sample_frames = []
            
            for idx, checkpoint_path in enumerate(tqdm(checkpoint_files, desc=f"Processing sample {sample_idx + 1}")):
                print(f"\n[DEBUG] Processing checkpoint {idx+1}/{len(checkpoint_files)}: {checkpoint_path.name}")
                
                # Load model
                print(f"[DEBUG] Loading model from {checkpoint_path.name}...")
                model, checkpoint = load_model_from_checkpoint(
                    str(checkpoint_path), config_dict, device
                )
                print(f"[DEBUG] Model loaded successfully")
                
                # Get training info
                epoch = checkpoint.get('epoch', 0)
                training_bytes = checkpoint.get('cumulative_training_bytes', 0)
                bytes_threshold = checkpoint.get('bytes_threshold', None)
                print(f"[DEBUG] Epoch: {epoch}, Training bytes: {training_bytes:,}, Bytes threshold: {bytes_threshold}")
                
                # Process this one sample
                print(f"[DEBUG] Processing SMILES: {sample_text[:50]}...")
                
                # Get boundary predictions (including all stages for 2-stage models)
                boundary_mask, boundary_prob, _, stages_data = get_boundary_predictions(
                    model, tokenizer, sample_text, device
                )
                
                num_stages = len(stages_data)
                texts_data = [{
                    'text': sample_text,
                    'boundary_mask': boundary_mask,
                    'boundary_prob': boundary_prob,
                    'stages': stages_data if num_stages > 1 else None,
                }]
                
                print(f"[DEBUG] Creating visualization frame ({num_stages} stages)...")
                
                # Create single frame with this sample (wrapped across rows)
                fig = create_multi_row_frame(
                    texts_data=texts_data,
                    epoch=epoch,
                    training_bytes=training_bytes,
                    is_concatenated=is_concatenated,
                    num_stages=num_stages,
                )
                
                print(f"[DEBUG] Frame created. Converting to image...")
                
                # Convert to image
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                img = np.asarray(buf)
                sample_frames.append(img)
                
                plt.close(fig)
                print(f"[DEBUG] Frame {idx+1} completed for sample {sample_idx + 1}")
            
            # Add final frame a few times for pause
            print("[DEBUG] Adding pause frames...")
            for _ in range(5):
                if sample_frames:
                    sample_frames.append(sample_frames[-1])
            
            # Save this sample as a separate GIF
            sample_output_path = output_dir / f"{output_base}_batch_{sample_idx + 1}.gif"
            print(f"\n[DEBUG] Saving GIF for sample {sample_idx + 1} to {sample_output_path}...")
            imageio.mimsave(sample_output_path, sample_frames, duration=0.5, loop=0)
            print(f"[DEBUG] GIF saved successfully!")
            print(f"Saved {len(sample_frames)} frames to {sample_output_path}")
        
        print(f"\n✓ Generated {len(texts)} GIFs successfully!")
    else:
        # Non-concatenated: create one GIF with all samples
        all_frames = []
        
        print(f"\n[DEBUG] Starting to process {len(checkpoint_files)} checkpoints...")
        for idx, checkpoint_path in enumerate(tqdm(checkpoint_files, desc="Processing checkpoints")):
            print(f"\n[DEBUG] Processing checkpoint {idx+1}/{len(checkpoint_files)}: {checkpoint_path.name}")
            
            # Load model
            print(f"[DEBUG] Loading model from {checkpoint_path.name}...")
            model, checkpoint = load_model_from_checkpoint(
                str(checkpoint_path), config_dict, device
            )
            print(f"[DEBUG] Model loaded successfully")
            
            # Get training info
            epoch = checkpoint.get('epoch', 0)
            training_bytes = checkpoint.get('cumulative_training_bytes', 0)
            bytes_threshold = checkpoint.get('bytes_threshold', None)
            print(f"[DEBUG] Epoch: {epoch}, Training bytes: {training_bytes:,}, Bytes threshold: {bytes_threshold}")
            
            # Process all texts for this checkpoint
            print(f"[DEBUG] Processing {len(texts)} SMILES strings...")
            texts_data = []
            num_stages = 1  # Will be updated from first prediction
            
            for text_idx, text in enumerate(texts):
                if text_idx % 10 == 0:  # Print every 10th to avoid spam
                    print(f"[DEBUG]   Processing SMILES {text_idx+1}/{len(texts)}: {text[:50]}...")
                
                # Get boundary predictions (including all stages for 2-stage models)
                boundary_mask, boundary_prob, _, stages_data = get_boundary_predictions(
                    model, tokenizer, text, device
                )
                
                num_stages = len(stages_data)
                texts_data.append({
                    'text': text,
                    'boundary_mask': boundary_mask,
                    'boundary_prob': boundary_prob,
                    'stages': stages_data if num_stages > 1 else None,
                })
            
            print(f"[DEBUG] All SMILES processed. Creating visualization frame ({num_stages} stages)...")
            
            # Create single frame with all texts in rows
            fig = create_multi_row_frame(
                texts_data=texts_data,
                epoch=epoch,
                training_bytes=training_bytes,
                is_concatenated=is_concatenated,
                num_stages=num_stages,
            )
            
            print(f"[DEBUG] Frame created. Converting to image...")
            
            # Convert to image
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)
            all_frames.append(img)
            
            plt.close(fig)
            print(f"[DEBUG] Frame {idx+1} completed and added to frames list (total: {len(all_frames)})")
        
        print(f"\n[DEBUG] All checkpoints processed. Total frames: {len(all_frames)}")
        
        # Add final frame a few times for pause
        print("[DEBUG] Adding pause frames...")
        for _ in range(5):
            if all_frames:
                all_frames.append(all_frames[-1])
        print(f"[DEBUG] Total frames after pause: {len(all_frames)}")
        
        # Save as GIF
        print(f"\n[DEBUG] Saving GIF to {output_path}...")
        print(f"[DEBUG] Output directory exists: {Path(output_path).parent.exists()}")
        print(f"[DEBUG] Number of frames to save: {len(all_frames)}")
        print(f"[DEBUG] Frame shape: {all_frames[0].shape if all_frames else 'N/A'}")
        
        imageio.mimsave(output_path, all_frames, duration=0.5, loop=0)
        print(f"[DEBUG] GIF saved successfully!")
        print(f"Saved {len(all_frames)} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize chunking evolution across training')
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to training run directory (contains checkpoints and metadata.json)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of SMILES strings to visualize from training dataset (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output GIF path (default: run_dir/visualizations/training_evolution.gif)'
    )
    parser.add_argument(
        '--checkpoint-pattern',
        type=str,
        default='checkpoint_bytes_*.pt',
        help='Pattern to match checkpoint files (default: checkpoint_bytes_*.pt for byte-based checkpoints)'
    )
    parser.add_argument(
        '--max-checkpoints',
        type=int,
        default=None,
        help='Maximum number of checkpoints to process (evenly spaced). None = all checkpoints'
    )
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = run_dir / "visualizations" / "training_evolution.gif"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create visualization (loads SMILES from dataset)
    visualize_training_evolution(
        run_dir=run_dir,
        num_samples=args.num_samples,
        output_path=str(output_path),
        checkpoint_pattern=args.checkpoint_pattern,
        max_checkpoints=args.max_checkpoints,
    )


if __name__ == '__main__':
    main()

