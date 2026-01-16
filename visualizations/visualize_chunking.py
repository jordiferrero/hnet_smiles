#!/usr/bin/env python3
"""
Visualize dynamic chunking on SMILES strings.
Creates animated GIFs showing how chunk boundaries evolve as tokens are processed.
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import imageio
from PIL import Image
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.utils.tokenizers import ByteTokenizer
from visualizations.utils import (
    bytes_to_hex,
    create_animation_frame,
    get_chunk_spans,
    draw_multistage_chunking_visualization,
)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load config
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Remove training_config if present (not part of model config)
    config_dict.pop("training_config", None)
    
    attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
    
    # Create model
    dtype = torch.bfloat16 if device.type != 'cpu' else torch.float32
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully!")
    return model


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


def visualize_progressive_chunking(
    model: HNetForCausalLM,
    tokenizer: ByteTokenizer,
    text: str,
    device: torch.device,
    output_path: str,
    frame_step: int = 1,
):
    """
    Create animated GIF showing progressive chunking as tokens are added.
    Supports both 1-stage and 2-stage H-Net models.
    """
    print(f"Visualizing progressive chunking for: {text[:50]}...")
    
    # Get full boundary predictions (including all stages)
    boundary_mask, boundary_prob, _, stages_data = get_boundary_predictions(
        model, tokenizer, text, device
    )
    
    num_stages = len(stages_data)
    print(f"Model has {num_stages} chunking stage(s)")
    
    # Create frames for progressive visualization
    frames = []
    hex_encoding = bytes_to_hex(text)
    
    # Process incrementally
    for length in range(1, len(text) + 1, frame_step):
        # Create frame with all stages
        fig = create_animation_frame(
            text=text,
            hex_encoding=hex_encoding,
            boundary_mask=boundary_mask,  # Stage 0 for backward compat
            boundary_prob=boundary_prob,
            current_length=length,
            frame_num=len(frames),
            stages_data=stages_data if num_stages > 1 else None,  # Pass all stages if multi-stage
        )
        
        # Convert to image
        fig.canvas.draw()
        # Use buffer_rgba() for newer matplotlib versions
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        frames.append(img)
        
        plt.close(fig)
    
    # Add final frame a few times for pause
    for _ in range(5):
        frames.append(frames[-1])
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, duration=0.2, loop=0)
    print(f"Saved {len(frames)} frames to {output_path}")


def visualize_batch(
    model: HNetForCausalLM,
    tokenizer: ByteTokenizer,
    texts: list,
    device: torch.device,
    output_dir: Path,
    prefix: str = "chunking",
):
    """Visualize chunking for multiple texts (creates individual GIFs)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, text in enumerate(tqdm(texts, desc="Visualizing")):
        output_path = output_dir / f"{prefix}_{i:03d}.gif"
        
        try:
            visualize_progressive_chunking(
                model, tokenizer, text, device, str(output_path)
            )
        except Exception as e:
            print(f"Error visualizing '{text[:50]}...': {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Visualize dynamic chunking')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config JSON'
    )
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Single SMILES string to visualize'
    )
    parser.add_argument(
        '--text-file',
        type=str,
        default=None,
        help='File with SMILES strings (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='visualizations/output',
        help='Output directory or file path'
    )
    parser.add_argument(
        '--frame-step',
        type=int,
        default=1,
        help='Number of characters per frame'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to visualize (if using text-file)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, args.config, device)
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Get texts to visualize
    texts = []
    if args.text:
        texts = [args.text]
    elif args.text_file:
        with open(args.text_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        texts = texts[:args.num_samples]
    else:
        # Default examples
        texts = [
            "*CCC[Fe]CCCC(=O)OCCCCOCCCNCC(*)=O",
            "*CCCC1C=CNC2=CC=C2C(*)CCC1",
            "*C(=O)CNC(*)C(=O)OCCCCCNC",
        ]
    
    # Visualize
    output_path = Path(args.output)
    if len(texts) == 1:
        # Single text: save to specified path
        if output_path.suffix != '.gif':
            output_path = output_path.with_suffix('.gif')
        visualize_progressive_chunking(
            model, tokenizer, texts[0], device, str(output_path), args.frame_step
        )
    else:
        # Multiple texts: save to directory
        visualize_batch(
            model, tokenizer, texts, device, output_path, prefix="chunking"
        )


if __name__ == '__main__':
    main()

