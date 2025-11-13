#!/usr/bin/env python3
"""
Visualize training statistics and chunking statistics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import sys
from collections import defaultdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.utils.tokenizers import ByteTokenizer


def format_bytes(bytes_val):
    """Format bytes to human-readable format."""
    if bytes_val >= 1_000_000_000:
        return f"{bytes_val/1_000_000_000:.1f}B"
    elif bytes_val >= 1_000_000:
        return f"{bytes_val/1_000_000:.1f}M"
    elif bytes_val >= 1_000:
        return f"{bytes_val/1_000:.1f}K"
    else:
        return str(bytes_val)


def plot_training_curves(checkpoint_dir: Path, output_path: Path, metadata_path: Path = None):
    """Plot training curves from checkpoints, using training bytes as x-axis."""
    # Try to load from metadata first (more reliable)
    if metadata_path and metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if 'training_history' in metadata and metadata['training_history']:
            # Collect all checkpoints (both epoch and byte-based) sorted by training bytes
            all_history = []
            for h in metadata['training_history']:
                cumulative_bytes = h.get('cumulative_training_bytes', 0)
                if cumulative_bytes > 0:  # Only include entries with training bytes
                    all_history.append({
                        'cumulative_bytes': cumulative_bytes,
                        'loss': h['metrics']['loss'],
                        'ce_loss': h['metrics']['ce_loss'],
                        'lb_loss': h['metrics']['lb_loss'],
                        'checkpoint_type': h.get('checkpoint_type', 'epoch'),
                    })
            
            # Sort by cumulative bytes
            all_history.sort(key=lambda x: x['cumulative_bytes'])
            
            if all_history:
                training_bytes = [h['cumulative_bytes'] for h in all_history]
                losses = [h['loss'] for h in all_history]
                ce_losses = [h['ce_loss'] for h in all_history]
                lb_losses = [h['lb_loss'] for h in all_history]
                
                # Plot from metadata
                fig, axes = plt.subplots(3, 1, figsize=(12, 10))
                
                axes[0].plot(training_bytes, losses, 'b-', marker='o', markersize=3, label='Total Loss')
                axes[0].set_xlabel('Training Bytes')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss vs Training Bytes')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_bytes(x)))
                
                axes[1].plot(training_bytes, ce_losses, 'r-', marker='o', markersize=3, label='Cross-Entropy Loss')
                axes[1].set_xlabel('Training Bytes')
                axes[1].set_ylabel('CE Loss')
                axes[1].set_title('Cross-Entropy Loss vs Training Bytes')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_bytes(x)))
                
                axes[2].plot(training_bytes, lb_losses, 'g-', marker='o', markersize=3, label='Load Balancing Loss')
                axes[2].set_xlabel('Training Bytes')
                axes[2].set_ylabel('LB Loss')
                axes[2].set_title('Load Balancing Loss vs Training Bytes')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_bytes(x)))
                
                plt.tight_layout()
                plt.savefig(output_path, dpi=150)
                print(f"Saved training curves to {output_path}")
                return
    
    # Fallback: collect metrics from checkpoints
    checkpoint_data = []
    
    for checkpoint_file in sorted(checkpoint_dir.glob("checkpoint_*.pt")):
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            if 'metrics' in checkpoint:
                cumulative_bytes = checkpoint.get('cumulative_training_bytes', 0)
                if cumulative_bytes > 0:
                    metrics = checkpoint['metrics']
                    checkpoint_data.append({
                        'cumulative_bytes': cumulative_bytes,
                        'loss': metrics.get('loss', 0),
                        'ce_loss': metrics.get('ce_loss', 0),
                        'lb_loss': metrics.get('lb_loss', 0),
                    })
        except Exception as e:
            print(f"Error loading {checkpoint_file}: {e}")
            continue
    
    if not checkpoint_data:
        print("No checkpoints found with metrics")
        return
    
    # Sort by cumulative bytes
    checkpoint_data.sort(key=lambda x: x['cumulative_bytes'])
    
    training_bytes = [d['cumulative_bytes'] for d in checkpoint_data]
    losses = [d['loss'] for d in checkpoint_data]
    ce_losses = [d['ce_loss'] for d in checkpoint_data]
    lb_losses = [d['lb_loss'] for d in checkpoint_data]
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(training_bytes, losses, 'b-', marker='o', markersize=3, label='Total Loss')
    axes[0].set_xlabel('Training Bytes')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss vs Training Bytes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_bytes(x)))
    
    axes[1].plot(training_bytes, ce_losses, 'r-', marker='o', markersize=3, label='Cross-Entropy Loss')
    axes[1].set_xlabel('Training Bytes')
    axes[1].set_ylabel('CE Loss')
    axes[1].set_title('Cross-Entropy Loss vs Training Bytes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_bytes(x)))
    
    axes[2].plot(training_bytes, lb_losses, 'g-', marker='o', markersize=3, label='Load Balancing Loss')
    axes[2].set_xlabel('Training Bytes')
    axes[2].set_ylabel('LB Loss')
    axes[2].set_title('Load Balancing Loss vs Training Bytes')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_bytes(x)))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved training curves to {output_path}")


def analyze_chunking_statistics(
    model: HNetForCausalLM,
    tokenizer: ByteTokenizer,
    texts: list,
    device: torch.device,
) -> dict:
    """Analyze chunking statistics on a set of texts."""
    chunk_sizes = []
    boundary_probs = []
    
    for text in texts:
        # Tokenize
        encoded = tokenizer.encode([text], add_bos=True, add_eos=True)[0]
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
            output = model.forward(input_ids, mask=mask)
            
            bpred_outputs = output.bpred_output
            if bpred_outputs and len(bpred_outputs) > 0:
                bpred = bpred_outputs[0]
                boundary_mask = bpred.boundary_mask[0].cpu().numpy()
                boundary_prob = bpred.boundary_prob[0, :, -1].cpu().float().numpy()  # Tokenized prob, convert to float32
                
                # Compute chunk sizes
                chunks = []
                start = 0
                for i, is_boundary in enumerate(boundary_mask):
                    if is_boundary and i > start:
                        chunks.append(i - start)
                        start = i
                if start < len(boundary_mask):
                    chunks.append(len(boundary_mask) - start)
                
                chunk_sizes.extend(chunks)
                boundary_probs.extend(boundary_prob.tolist())
    
    return {
        'chunk_sizes': chunk_sizes,
        'boundary_probs': boundary_probs,
    }


def plot_chunking_statistics(stats: dict, output_path: Path):
    """Plot chunking statistics."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Chunk size distribution
    axes[0].hist(stats['chunk_sizes'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Chunk Size')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Chunk Size Distribution')
    axes[0].axvline(np.mean(stats['chunk_sizes']), color='r', linestyle='--',
                    label=f'Mean: {np.mean(stats["chunk_sizes"]):.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boundary probability distribution
    axes[1].hist(stats['boundary_probs'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Boundary Probability')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Boundary Probability Distribution')
    axes[1].axvline(np.mean(stats['boundary_probs']), color='r', linestyle='--',
                    label=f'Mean: {np.mean(stats["boundary_probs"]):.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved chunking statistics to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training and chunking statistics')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (for chunking stats)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model config JSON'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory with checkpoints (for training curves). If not provided, will use run-dir/checkpoints'
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help='Path to training run directory (contains checkpoints and metadata.json). Takes precedence over checkpoint-dir'
    )
    parser.add_argument(
        '--text-file',
        type=str,
        default=None,
        help='File with SMILES strings for chunking analysis'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations/stats',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine checkpoint directory and metadata path
    if args.run_dir:
        run_dir = Path(args.run_dir)
        checkpoint_dir = run_dir / "checkpoints"
        metadata_path = run_dir / "metadata.json"
    elif args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        metadata_path = None
    else:
        checkpoint_dir = Path('checkpoints')
        metadata_path = None
    
    # Plot training curves if checkpoint directory exists
    if checkpoint_dir.exists():
        plot_training_curves(
            checkpoint_dir,
            output_dir / 'training_curves.png',
            metadata_path=metadata_path
        )
    
    # Analyze chunking statistics if model is provided
    if args.checkpoint and args.config:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                            "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load model
        print("Loading model...")
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
        ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
        hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
        
        dtype = torch.bfloat16 if device.type != 'cpu' else torch.float32
        model = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
        model.eval()
        
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Get texts
        if args.text_file:
            with open(args.text_file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            # Default samples
            texts = [
                "*CCC[Fe]CCCC(=O)OCCCCOCCCNCC(*)=O",
                "*CCCC1C=CNC2=CC=C2C(*)CCC1",
                "*C(=O)CNC(*)C(=O)OCCCCCNC",
            ] * 10  # Repeat for more statistics
        
        tokenizer = ByteTokenizer()
        
        print("Analyzing chunking statistics...")
        stats = analyze_chunking_statistics(model, tokenizer, texts, device)
        
        plot_chunking_statistics(stats, output_dir / 'chunking_statistics.png')
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()

