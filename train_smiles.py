#!/usr/bin/env python3
"""
Training script for HNet on SMILES data.
Supports incremental training phases: small -> medium -> large
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import json
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.utils.tokenizers import ByteTokenizer
from hnet.utils.train import load_balancing_loss, group_params
from data.smiles_dataset import create_dataloader


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> HNetConfig:
    """Load model configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
    
    return hnet_cfg


def create_model(config: HNetConfig, device, dtype=torch.bfloat16):
    """Create and initialize HNet model."""
    model = HNetForCausalLM(config, device=device, dtype=dtype)
    model.init_weights(initializer_range=0.02)
    
    # Apply learning rate multipliers if needed
    # lr_multiplier = [3.0, 1.7, 0.9]  # Example, adjust based on config
    # model.apply_lr_multiplier(lr_multiplier)
    
    return model


def compute_loss(
    model: HNetForCausalLM,
    batch: dict,
    device: torch.device,
    use_amp: bool = True,
):
    """
    Compute training loss including:
    - Cross-entropy loss for next-token prediction
    - Load balancing loss from routing module
    """
    # Prepare inputs based on batching format
    if 'cu_seqlens' in batch:
        # Packed format: input_ids is already flattened
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        cu_seqlens = batch['cu_seqlens'].to(device)
        max_seqlen = None
        mask = None
    else:
        # Padded format
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        cu_seqlens = None
        max_seqlen = None
    
    with autocast(enabled=use_amp, dtype=torch.bfloat16):
        # Forward pass
        if cu_seqlens is not None:
            # Packed mode: input_ids is (T,) flattened
            # Model expects (B, L, D) for embeddings, but works with packed
            # We need to reshape input_ids to (1, T) for embedding, then flatten
            output = model(
                input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=None,
            )
        else:
            # Padded mode
            output = model(
                input_ids,
                mask=mask,
            )
        
        logits = output.logits
        bpred_outputs = output.bpred_output
    
    # Compute cross-entropy loss
    if cu_seqlens is not None:
        # Packed format: logits are reshaped to (B, L, V) but need to match labels
        # Labels are already properly aligned with input_ids
        if logits.dim() == 3:
            # Reshape to match labels
            logits_flat = logits.view(-1, logits.size(-1))
        else:
            logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
    else:
        # Padded format
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        mask_flat = mask.view(-1)
        
        # Mask out padding tokens
        valid_mask = mask_flat.bool()
        logits_flat = logits_flat[valid_mask]
        labels_flat = labels_flat[valid_mask]
    
    ce_loss = nn.functional.cross_entropy(logits_flat, labels_flat)
    
    # Compute load balancing loss
    lb_loss = torch.tensor(0.0, device=device)
    if bpred_outputs:
        # Average load balancing loss across all stages
        for bpred in bpred_outputs:
            if bpred is not None:
                # N is the downsampling factor (approximate from model architecture)
                N = 2.0  # Typical downsampling factor
                lb_loss += load_balancing_loss(bpred, N)
        lb_loss = lb_loss / len(bpred_outputs) if bpred_outputs else lb_loss
    
    # Total loss
    total_loss = ce_loss + 0.01 * lb_loss  # Weight load balancing loss
    
    return {
        'total_loss': total_loss,
        'ce_loss': ce_loss,
        'lb_loss': lb_loss,
    }


def train_epoch(
    model: HNetForCausalLM,
    dataloader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_lb_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        # Compute loss
        loss_dict = compute_loss(model, batch, device, use_amp)
        
        # Scale loss for gradient accumulation
        loss = loss_dict['total_loss'] / gradient_accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += loss_dict['total_loss'].item()
        total_ce_loss += loss_dict['ce_loss'].item()
        total_lb_loss += loss_dict['lb_loss'].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{total_loss/num_batches:.4f}",
            'ce': f"{total_ce_loss/num_batches:.4f}",
            'lb': f"{total_lb_loss/num_batches:.6f}",
        })
    
    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'lb_loss': total_lb_loss / num_batches,
    }


def validate(
    model: HNetForCausalLM,
    dataloader,
    device: torch.device,
    use_amp: bool = True,
):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_lb_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            loss_dict = compute_loss(model, batch, device, use_amp)
            
            total_loss += loss_dict['total_loss'].item()
            total_ce_loss += loss_dict['ce_loss'].item()
            total_lb_loss += loss_dict['lb_loss'].item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'lb_loss': total_lb_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train HNet on SMILES data')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config JSON file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='datasets/PI1M/PI1M_v2.csv',
        help='Path to SMILES CSV file'
    )
    parser.add_argument(
        '--phase',
        type=str,
        choices=['small', 'medium', 'large'],
        default='small',
        help='Training phase (small/medium/large)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to use (None = all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.1,
        help='Weight decay'
    )
    parser.add_argument(
        '--gradient-accumulation',
        type=int,
        default=1,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkpoints',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable automatic mixed precision'
    )
    parser.add_argument(
        '--concatenate',
        action='store_true',
        default=True,
        help='Concatenate multiple SMILES'
    )
    parser.add_argument(
        '--num-concatenate',
        type=int,
        default=3,
        help='Number of SMILES to concatenate'
    )
    
    args = parser.parse_args()
    
    # Set phase-specific defaults
    phase_configs = {
        'small': {'max_samples': 1000, 'epochs': 5, 'batch_size': 8},
        'medium': {'max_samples': 10000, 'epochs': 10, 'batch_size': 16},
        'large': {'max_samples': None, 'epochs': 20, 'batch_size': 32},
    }
    
    if args.max_samples is None:
        args.max_samples = phase_configs[args.phase]['max_samples']
    if args.epochs == 10 and args.phase in phase_configs:
        args.epochs = phase_configs[args.phase]['epochs']
    if args.batch_size == 8 and args.phase in phase_configs:
        args.batch_size = phase_configs[args.phase]['batch_size']
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load config
    print(f"Loading config from {args.config}...")
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    
    attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
    
    # Create model
    print("Creating model...")
    dtype = torch.bfloat16 if device.type != 'cpu' else torch.float32
    model = create_model(hnet_cfg, device, dtype=dtype)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Create tokenizer and dataloader
    tokenizer = ByteTokenizer()
    
    print(f"Creating dataloader (phase={args.phase}, max_samples={args.max_samples})...")
    train_loader = create_dataloader(
        csv_path=args.data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        concatenate=args.concatenate,
        num_concatenate=args.num_concatenate,
        shuffle=True,
    )
    
    # Create optimizer
    param_groups = group_params(model)
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    # Mixed precision
    use_amp = not args.no_amp and device.type != 'cpu'
    scaler = GradScaler() if use_amp else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training (phase={args.phase})...")
    print(f"Device: {device}, AMP: {use_amp}, Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.lr}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            args.gradient_accumulation,
        )
        
        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"(CE: {train_metrics['ce_loss']:.4f}, "
              f"LB: {train_metrics['lb_loss']:.6f})")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_phase_{args.phase}_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config_dict,
            'metrics': train_metrics,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

