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

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.utils.tokenizers import ByteTokenizer
from visualizations.utils import (
    bytes_to_hex,
    create_animation_frame,
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


def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load config
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
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
    
    Returns:
        (boundary_mask, boundary_prob, text)
    """
    # Tokenize
    encoded = tokenizer.encode([text], add_bos=True, add_eos=True)[0]
    input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        output = model.forward(input_ids, mask=mask)
        
        # Extract boundary predictions from first stage
        bpred_outputs = output.bpred_output
        if bpred_outputs and len(bpred_outputs) > 0:
            bpred = bpred_outputs[0]  # First stage
            boundary_mask = bpred.boundary_mask[0].cpu().numpy()  # (L,)
            boundary_prob = bpred.boundary_prob[0].cpu().numpy()  # (L, 2)
        else:
            # Fallback: no boundaries detected
            boundary_mask = np.zeros(len(encoded['input_ids']), dtype=bool)
            boundary_mask[0] = True  # First token is always a boundary
            boundary_prob = np.zeros((len(encoded['input_ids']), 2))
    
    # Remove BOS/EOS tokens for visualization
    # BOS is at position 0, EOS is at the end
    boundary_mask = boundary_mask[1:-1]  # Remove BOS and EOS
    boundary_prob = boundary_prob[1:-1]
    
    return boundary_mask, boundary_prob, text


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
    """
    print(f"Visualizing progressive chunking for: {text[:50]}...")
    
    # Get full boundary predictions
    boundary_mask, boundary_prob, _ = get_boundary_predictions(
        model, tokenizer, text, device
    )
    
    # Create frames for progressive visualization
    frames = []
    hex_encoding = bytes_to_hex(text)
    
    # Process incrementally
    for length in range(1, len(text) + 1, frame_step):
        # Get boundary mask up to current length
        current_mask = boundary_mask[:length]
        
        # Create frame
        fig = create_animation_frame(
            text=text,
            hex_encoding=hex_encoding,
            boundary_mask=boundary_mask,  # Full mask for context
            boundary_prob=boundary_prob,
            current_length=length,
            frame_num=len(frames),
        )
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
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
    """Visualize chunking for multiple texts."""
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

