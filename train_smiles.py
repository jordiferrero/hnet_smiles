#!/usr/bin/env python3
"""
Training script for HNet on SMILES data.
Supports incremental training phases: small -> medium -> large
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import json
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from datetime import datetime
import shutil
import signal
import os
import pickle
import gzip

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.utils.tokenizers import ByteTokenizer
from hnet.utils.train import load_balancing_loss, group_params
from data.smiles_dataset import create_dataloader, SMILESDataset


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle signals gracefully."""
    global shutdown_requested
    if signum == signal.SIGHUP:
        # Ignore SIGHUP (SSH disconnection) - continue training
        # Note: This helps but for full SSH disconnection protection, use nohup, screen, tmux, or systemd service
        print(f"\nReceived SIGHUP (SSH disconnection). Continuing training...")
        return
    elif signum in (signal.SIGTERM, signal.SIGINT):
        # Request graceful shutdown on SIGTERM/SIGINT
        print(f"\nReceived signal {signum}. Requesting graceful shutdown after current batch...")
        shutdown_requested = True


def setup_signal_handlers():
    """
    Setup signal handlers to survive SSH disconnections.
    
    Note: Signal handling alone is not sufficient for SSH disconnections.
    For proper SSH disconnection protection, use one of:
    - nohup: nohup python train_smiles.py ... &
    - screen: screen -S training, then run script, detach with Ctrl+A D
    - tmux: tmux new -s training, then run script, detach with Ctrl+B D
    - systemd service: Create a service file and run as systemd service
    
    Signal handling here provides a safety net but the process should be
    detached from the terminal for full protection.
    """
    # Ignore SIGHUP (SSH disconnection)
    signal.signal(signal.SIGHUP, signal_handler)
    # Handle SIGTERM and SIGINT gracefully
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    print("Signal handlers configured (for full SSH protection, use nohup/screen/tmux/service).")


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
    
    # Apply learning rate modulation (Section 2.3 of paper)
    # For 1-stage H-Net: [encoder/decoder LR, main network LR]
    # Outer stages (encoder/decoder) get higher LR, inner stages get lower LR
    # Determine number of stages from arch_layout
    arch_layout = config.arch_layout
    num_stages = 1
    while isinstance(arch_layout, list) and len(arch_layout) > 1:
        if len(arch_layout) == 3:  # encoder, main, decoder
            arch_layout = arch_layout[1]  # main network
            num_stages += 1
        else:
            break
    
    # Set LR multipliers: outer stages get higher LR (3.0), inner stages get lower LR (0.9)
    # For 1-stage: [3.0, 0.9] (encoder/decoder, main)
    # For 2-stage: [3.0, 1.7, 0.9] (outer encoder/decoder, middle, inner main)
    if num_stages == 1:
        lr_multiplier = [3.0, 0.9]
    elif num_stages == 2:
        lr_multiplier = [3.0, 1.7, 0.9]
    else:
        # Default: linearly interpolate between 3.0 and 0.9
        lr_multiplier = [3.0 - (3.0 - 0.9) * i / (num_stages - 1) for i in range(num_stages + 1)]
    
    model.apply_lr_multiplier(lr_multiplier)
    print(f"Applied learning rate multipliers: {lr_multiplier} (for {num_stages} stages)")
    
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
    # Prepare inputs - using padded format
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    mask = batch['mask'].to(device)
    
    with autocast(device_type='cuda', enabled=use_amp, dtype=torch.bfloat16):
        # Forward pass
        output = model(
            input_ids,
            mask=mask,
        )
        
        logits = output.logits
        bpred_outputs = output.bpred_output
    
    # Compute cross-entropy loss
    # Padded format: mask out padding tokens
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
                # Calculate actual downsampling factor N from the batch
                # N = total_tokens / selected_tokens (compression ratio)
                total_tokens = bpred.boundary_mask.numel()
                selected_tokens = bpred.boundary_mask.sum().item()
                if selected_tokens > 0:
                    N = total_tokens / selected_tokens
                else:
                    N = 2.0  # Fallback if no tokens selected
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
    checkpoint_callback=None,
    metrics_accumulator=None,
):
    """
    Train for one epoch. Returns metrics and training bytes.
    
    Args:
        checkpoint_callback: Optional callback function(bytes_processed, metrics_dict) called during training
                             to save checkpoints based on bytes processed.
        metrics_accumulator: Optional dict to accumulate metrics during training.
    """
    global shutdown_requested
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_lb_loss = 0.0
    num_batches = 0
    training_bytes = 0  # Track training bytes
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        # Check for shutdown request
        if shutdown_requested:
            print("\nShutdown requested. Saving checkpoint and exiting...")
            break
        
        # Count training bytes (characters in input sequences)
        batch_bytes = 0
        if 'mask' in batch:
            # Padded format: count non-padding tokens
            valid_tokens = batch['mask'].sum().item()
            batch_bytes = valid_tokens
            training_bytes += batch_bytes
        elif 'cu_seqlens' in batch:
            # Packed format: all tokens are valid
            batch_bytes = batch['input_ids'].numel()
            training_bytes += batch_bytes
        
        # Compute loss
        loss_dict = compute_loss(model, batch, device, use_amp)
        
        # Scale loss for gradient accumulation
        loss = loss_dict['total_loss'] / gradient_accumulation_steps
        
        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None:
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
        
        # Update metrics accumulator if provided
        if metrics_accumulator is not None:
            metrics_accumulator['total_loss'] += loss_dict['total_loss'].item()
            metrics_accumulator['total_ce_loss'] += loss_dict['ce_loss'].item()
            metrics_accumulator['total_lb_loss'] += loss_dict['lb_loss'].item()
            metrics_accumulator['num_batches'] += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{total_loss/num_batches:.4f}",
            'ce': f"{total_ce_loss/num_batches:.4f}",
            'lb': f"{total_lb_loss/num_batches:.6f}",
            'bytes': f"{training_bytes:,}",
        })
        
        # Call checkpoint callback if provided
        if checkpoint_callback is not None:
            current_metrics = {
                'loss': total_loss / num_batches if num_batches > 0 else 0.0,
                'ce_loss': total_ce_loss / num_batches if num_batches > 0 else 0.0,
                'lb_loss': total_lb_loss / num_batches if num_batches > 0 else 0.0,
            }
            checkpoint_callback(training_bytes, current_metrics)
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'ce_loss': total_ce_loss / num_batches if num_batches > 0 else 0.0,
        'lb_loss': total_lb_loss / num_batches if num_batches > 0 else 0.0,
        'training_bytes': training_bytes,
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


def save_boundary_predictions(
    model: HNetForCausalLM,
    tokenizer: ByteTokenizer,
    test_smiles: list,
    epoch: int,
    training_bytes: int,
    bytes_threshold: int,
    output_dir: Path,
    device: torch.device,
):
    """
    Save boundary predictions to compressed pickle files (one per checkpoint) for later visualization.
    Saves predictions for ALL test samples (num_test_samples).
    Uses pickle + gzip compression for efficient storage.
    
    Args:
        model: The model to use for predictions
        tokenizer: Tokenizer for encoding SMILES
        test_smiles: List of ALL test SMILES strings (num_test_samples)
        epoch: Current epoch number
        training_bytes: Current training bytes
        bytes_threshold: Byte threshold for this checkpoint
        output_dir: Directory to save prediction files
        device: Device to run on
    """
    model.eval()
    predictions = []
    
    # Save predictions for ALL test samples (not just num_visualize)
    with torch.no_grad():
        for text in test_smiles:
            # Tokenize
            encoded = tokenizer.encode([text], add_bos=True, add_eos=True)[0]
            input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
            
            # Forward pass
            mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
            output = model.forward(input_ids, mask=mask)
            
            # Extract boundary predictions from first stage
            bpred_outputs = output.bpred_output
            if bpred_outputs and len(bpred_outputs) > 0:
                bpred = bpred_outputs[0]  # First stage
                boundary_mask = bpred.boundary_mask[0].cpu().numpy()  # (L,)
                boundary_prob = bpred.boundary_prob[0].cpu().float().numpy()  # (L, 2)
            else:
                # Fallback: no boundaries detected
                boundary_mask = np.zeros(len(encoded['input_ids']), dtype=bool)
                boundary_mask[0] = True  # First token is always a boundary
                boundary_prob = np.zeros((len(encoded['input_ids']), 2))
            
            # Remove BOS/EOS tokens for visualization
            boundary_mask = boundary_mask[1:-1]  # Remove BOS and EOS
            boundary_prob = boundary_prob[1:-1]
            
            # Store as numpy arrays (more efficient than lists)
            predictions.append({
                'smiles': text,
                'boundary_mask': boundary_mask,  # Keep as numpy array
                'boundary_prob': boundary_prob,  # Keep as numpy array
            })
    
    # Create structured data entry
    data_entry = {
        'epoch': epoch,
        'training_bytes': training_bytes,
        'bytes_threshold': bytes_threshold,
        'predictions': predictions,
        'test_smiles': test_smiles,  # Store test SMILES for reference
    }
    
    # Save as compressed pickle file (one file per checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / f"predictions_bytes_{bytes_threshold:,}.pkl.gz"
    
    with gzip.open(checkpoint_file, 'wb') as f:
        pickle.dump(data_entry, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    model.train()  # Set back to training mode


def run_visualization(
    run_dir: Path,
    checkpoint_type: str,
    output_dir: Path,
    num_visualize: int = 5,
):
    """
    Run visualization on test SMILES after training using existing visualization scripts.
    
    Args:
        run_dir: Path to run directory (contains checkpoints, metadata.json, test_smiles.txt)
        checkpoint_type: Type of checkpoint ('epoch' or 'bytes')
        output_dir: Directory to save visualizations
        num_visualize: Number of SMILES to visualize
    """
    print("\n" + "="*60)
    print(f"Running visualization for {checkpoint_type} checkpoints...")
    print("="*60)
    
    try:
        import subprocess
        import sys
        
        # Use existing visualize_training_evolution.py script
        # It will automatically use test_smiles.txt if available, or load from dataset
        visualize_script = Path(__file__).parent / "visualizations" / "visualize_training_evolution.py"
        
        if not visualize_script.exists():
            print(f"Warning: Visualization script not found: {visualize_script}")
            return
        
        # Determine checkpoint pattern based on type
        if checkpoint_type == "epoch":
            checkpoint_pattern = "checkpoint_epoch_*.pt"
        elif checkpoint_type == "bytes":
            checkpoint_pattern = "checkpoint_bytes_best.pt"
        else:
            checkpoint_pattern = "checkpoint_*.pt"
        
        # Create output path
        output_path = output_dir / "chunking_combined.gif"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run visualization script
        print(f"Running visualization script: {visualize_script.name}")
        print(f"  Run directory: {run_dir}")
        print(f"  Checkpoint pattern: {checkpoint_pattern}")
        print(f"  Output: {output_path}")
        
        cmd = [
            sys.executable,
            str(visualize_script),
            "--run-dir", str(run_dir),
            "--num-samples", str(num_visualize),
            "--output", str(output_path),
            "--checkpoint-pattern", checkpoint_pattern,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n✓ Visualization complete! Saved to {output_path}")
        else:
            print(f"\n✗ Visualization failed:")
            print(result.stderr)
            print("\nContinuing without visualization...")
        
    except Exception as e:
        print(f"\n✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nContinuing without visualization...")


def main():
    parser = argparse.ArgumentParser(description='Train HNet on SMILES data')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config JSON file (contains all training parameters)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint (optional override)'
    )
    
    # Parse only config argument first to load config
    args, remaining_args = parser.parse_known_args()
    
    # Load config file
    print(f"Loading config from {args.config}...")
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    
    # Extract training_config
    training_config = config_dict.pop("training_config", {})
    
    # Set defaults from config file
    defaults = {
        'data': training_config.get('data', 'datasets/PI1M/PI1M_v2.csv'),
        'phase': training_config.get('phase', 'small'),
        'max_samples': training_config.get('max_samples', None),
        'batch_size': training_config.get('batch_size', 8),
        'epochs': training_config.get('epochs', 10),
        'lr': training_config.get('lr', 1e-4),
        'weight_decay': training_config.get('weight_decay', 0.1),
        'gradient_accumulation': training_config.get('gradient_accumulation', 1),
        'concatenate': training_config.get('concatenate', True),
        'num_concatenate': training_config.get('num_concatenate', 3),
        'concatenate_separator': training_config.get('concatenate_separator', ' '),
        'checkpoint_bytes': training_config.get('checkpoint_bytes', 1_000_000),
        'num_test_samples': training_config.get('num_test_samples', 50),
        'num_visualize': training_config.get('num_visualize', 5),
        'skip_visualization': training_config.get('skip_visualization', False),
        'output_dir': training_config.get('output_dir', 'checkpoints'),
        'no_amp': training_config.get('no_amp', False),
    }
    
    # Add all arguments with defaults from config
    parser.add_argument('--data', type=str, default=defaults['data'], help='Path to SMILES CSV file')
    parser.add_argument('--phase', type=str, choices=['small', 'medium', 'large'], default=defaults['phase'], help='Training phase')
    parser.add_argument('--max-samples', type=int, default=defaults['max_samples'], help='Maximum number of samples (None = all)')
    parser.add_argument('--batch-size', type=int, default=defaults['batch_size'], help='Batch size')
    parser.add_argument('--epochs', type=int, default=defaults['epochs'], help='Number of epochs')
    parser.add_argument('--lr', type=float, default=defaults['lr'], help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=defaults['weight_decay'], help='Weight decay')
    parser.add_argument('--gradient-accumulation', type=int, default=defaults['gradient_accumulation'], help='Gradient accumulation steps')
    parser.add_argument('--concatenate', action='store_true', default=defaults['concatenate'], help='Concatenate multiple SMILES')
    parser.add_argument('--no-concatenate', dest='concatenate', action='store_false', help='Disable concatenation')
    parser.add_argument('--num-concatenate', type=int, default=defaults['num_concatenate'], help='Number of SMILES to concatenate')
    parser.add_argument('--concatenate-separator', type=str, default=defaults['concatenate_separator'], help='Separator character(s) between concatenated SMILES (default: space)')
    parser.add_argument('--checkpoint-bytes', type=int, default=defaults['checkpoint_bytes'], help='Save checkpoint every N training bytes')
    parser.add_argument('--num-test-samples', type=int, default=defaults['num_test_samples'], help='Number of SMILES for testing')
    parser.add_argument('--num-visualize', type=int, default=defaults['num_visualize'], help='Number of test SMILES to visualize')
    parser.add_argument('--skip-visualization', action='store_true', default=defaults['skip_visualization'], help='Skip visualization')
    parser.add_argument('--output-dir', type=str, default=defaults['output_dir'], help='Output directory for checkpoints')
    parser.add_argument('--no-amp', action='store_true', default=defaults['no_amp'], help='Disable automatic mixed precision')
    
    # Parse all arguments (config values are defaults, cmd line can override)
    args = parser.parse_args()
    
    # Convert max_samples None/null to None
    if args.max_samples == "None" or args.max_samples == "null" or args.max_samples == "None":
        args.max_samples = None
    
    # Setup signal handlers (must be done early)
    setup_signal_handlers()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Config already loaded above, now extract model config
    # Save full config for later (before popping)
    full_config_dict = config_dict.copy()
    
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
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Split dataset into train/test for visualization
    print(f"Loading and splitting dataset (phase={args.phase}, max_samples={args.max_samples})...")
    
    # Load full dataset to split
    full_dataset = SMILESDataset(
        csv_path=args.data,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        concatenate=args.concatenate,
        num_concatenate=args.num_concatenate,
        separator=args.concatenate_separator,
    )
    
    # Split into train/test
    dataset_size = len(full_dataset)
    test_size = min(args.num_test_samples, dataset_size)  # Don't exceed dataset size
    train_size = dataset_size - test_size
    
    # Validate num_visualize
    if args.num_visualize > test_size:
        print(f"Warning: num_visualize ({args.num_visualize}) > num_test_samples ({test_size}). Setting num_visualize to {test_size}.")
        args.num_visualize = test_size
    
    # Use random split with fixed seed for reproducibility
    import random
    random.seed(42)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    # Take first test_size samples for test set
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # Create train dataset
    train_smiles = [full_dataset.smiles[i] for i in train_indices]
    train_dataset = SMILESDataset(
        csv_path=args.data,
        tokenizer=tokenizer,
        max_samples=None,  # We'll handle limiting manually
        concatenate=args.concatenate,
        num_concatenate=args.num_concatenate,
        separator=args.concatenate_separator,
    )
    train_dataset.smiles = train_smiles  # Override with split data
    
    # Create test dataset (for visualization)
    test_smiles = [full_dataset.smiles[i] for i in test_indices]
    
    print(f"Dataset split: {train_size:,} train, {test_size:,} test (fixed number of samples)")
    
    # Create timestamped run directory (needed for test_smiles_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{args.phase}_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save test SMILES to file for visualization
    test_smiles_file = run_dir / "test_smiles.txt"
    with open(test_smiles_file, 'w') as f:
        for smiles in test_smiles:
            f.write(f"{smiles}\n")
    print(f"Saved {len(test_smiles)} test SMILES to {test_smiles_file}")
    
    # Create train dataloader
    from torch.utils.data import DataLoader
    from data.smiles_dataset import collate_fn_padded
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_padded,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Create optimizer with hierarchical parameter groups (Section 2.3 of paper)
    # This respects the learning rate multipliers and weight decay settings
    # Note: group_params must be called AFTER apply_lr_multiplier
    param_groups = group_params(model)
    
    # Apply base learning rate and weight decay to all groups
    for group in param_groups:
        # Get the LR multiplier from the group (if set), otherwise use 1.0
        lr_mult = group.get('lr_multiplier', 1.0)
        group['lr'] = args.lr * lr_mult
        # Weight decay: group_params sets 0.0 for bias/norm parameters,
        # but doesn't set it for regular parameters. We need to set args.weight_decay for groups
        # that don't have weight_decay set (i.e., regular parameters).
        if 'weight_decay' not in group:
            group['weight_decay'] = args.weight_decay
    
    # Note: AdamW's lr parameter is ignored when param_groups have their own 'lr' keys
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    
    # Print parameter group info for debugging
    print(f"Created {len(param_groups)} parameter groups with learning rate modulation")
    for i, group in enumerate(param_groups):
        lr_mult = group.get('lr_multiplier', 1.0)
        num_params = sum(p.numel() for p in group['params'])
        wd = group.get('weight_decay', args.weight_decay)
        print(f"  Group {i}: {num_params:,} params, LR={group['lr']:.2e} (mult={lr_mult:.2f}), WD={wd}")
    
    # Mixed precision
    use_amp = not args.no_amp and device.type != 'cpu'
    # For bfloat16, we don't need a scaler (it's more stable than float16)
    scaler = None if dtype == torch.bfloat16 else (GradScaler('cuda') if use_amp else None)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    cumulative_training_bytes = 0
    last_checkpoint_bytes = 0  # Track last checkpoint byte threshold
    best_loss = None  # Track best loss for loss-based checkpoint saving
    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        cumulative_training_bytes = checkpoint.get('cumulative_training_bytes', 0)
        # Set last checkpoint bytes to the last checkpoint threshold
        last_checkpoint_bytes = (cumulative_training_bytes // args.checkpoint_bytes) * args.checkpoint_bytes
        # Restore best loss if available
        best_loss = checkpoint.get('best_loss', None)
    
    # Run directory already created above (for test_smiles_file)
    # Create subdirectories
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    visualizations_dir = run_dir / "visualizations"
    visualizations_dir.mkdir(exist_ok=True)
    
    # Save metadata for reproducibility
    metadata = {
        'run_name': run_name,
        'timestamp': timestamp,
        'phase': args.phase,
        'config': full_config_dict,
        'training_args': {
            'data': args.data,
            'max_samples': args.max_samples,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'gradient_accumulation': args.gradient_accumulation,
            'concatenate': args.concatenate,
            'num_concatenate': args.num_concatenate,
            'concatenate_separator': args.concatenate_separator,
            'checkpoint_bytes': args.checkpoint_bytes,
            'num_test_samples': args.num_test_samples,
            'num_visualize': args.num_visualize,
            'skip_visualization': args.skip_visualization,
        },
        'dataset_info': {
            'train_size': train_size,
            'test_size': test_size,
            'test_smiles_file': str(test_smiles_file),
        },
        'model_info': {
            'num_parameters': num_params,
            'device': str(device),
            'dtype': str(dtype),
            'use_amp': use_amp,
        },
        'training_history': [],
    }
    
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Training loop
    print(f"\nStarting training (phase={args.phase})...")
    print(f"Device: {device}, AMP: {use_amp}, Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.lr}")
    print(f"Checkpoint every {args.checkpoint_bytes:,} training bytes")
    
    # Track metrics for current epoch
    epoch_metrics_accumulator = {
        'total_loss': 0.0,
        'total_ce_loss': 0.0,
        'total_lb_loss': 0.0,
        'num_batches': 0,
    }
    
    def save_checkpoint(checkpoint_type, epoch=None, bytes_threshold=None, metrics=None, cumulative_bytes=None):
        """Save a checkpoint with appropriate naming."""
        if checkpoint_type == 'epoch':
            checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            checkpoint_name = f"epoch_{epoch + 1}"
        elif checkpoint_type == 'bytes':
            # Use fixed filename for bytes checkpoints - always replace with best one
            checkpoint_path = checkpoints_dir / "checkpoint_bytes_best.pt"
            checkpoint_name = f"bytes_best"
        else:
            checkpoint_path = checkpoints_dir / f"checkpoint_final.pt"
            checkpoint_name = "final"
        
        # Use provided cumulative_bytes or default to cumulative_training_bytes
        checkpoint_cumulative_bytes = cumulative_bytes if cumulative_bytes is not None else cumulative_training_bytes
        
        checkpoint_data = {
            'epoch': epoch if epoch is not None else start_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': full_config_dict,
            'metrics': metrics,
            'cumulative_training_bytes': checkpoint_cumulative_bytes,
            'checkpoint_type': checkpoint_type,
            'checkpoint_name': checkpoint_name,
            'best_loss': best_loss,  # Save best loss for resuming
        }
        if bytes_threshold is not None:
            checkpoint_data['bytes_threshold'] = bytes_threshold
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Saved {checkpoint_type} checkpoint to {checkpoint_path}")
            return checkpoint_path
        except RuntimeError as e:
            error_msg = str(e)
            if "file write failed" in error_msg or "unexpected pos" in error_msg:
                import shutil
                free_space = shutil.disk_usage(checkpoint_path.parent).free / (1024**3)  # GB
                raise RuntimeError(
                    f"Failed to save checkpoint: disk may be full or I/O error occurred. "
                    f"Available space: {free_space:.2f} GB. "
                    f"Original error: {error_msg}"
                ) from e
            else:
                raise
    
    for epoch in range(start_epoch, args.epochs):
        def checkpoint_callback(epoch_bytes_processed, current_metrics):
            """Callback function called during training to save predictions at every checkpoint_bytes interval, and model checkpoints only when loss improves."""
            nonlocal cumulative_training_bytes, last_checkpoint_bytes, best_loss
            
            # Calculate cumulative bytes for this epoch
            current_cumulative = cumulative_training_bytes + epoch_bytes_processed
            
            # Check if we've crossed a checkpoint threshold
            current_threshold = (current_cumulative // args.checkpoint_bytes) * args.checkpoint_bytes
            if current_threshold > last_checkpoint_bytes:
                current_loss = current_metrics.get('loss', float('inf'))
                
                # ALWAYS save boundary predictions at every checkpoint_bytes interval (regardless of loss)
                if not args.skip_visualization:
                    predictions_dir = visualizations_dir / "predictions"
                    
                    print(f"[PREDICTIONS] Saving at {current_threshold:,} bytes (loss: {current_loss:.4f}, best: {best_loss})...")
                    try:
                        save_boundary_predictions(
                            model=model,
                            tokenizer=tokenizer,
                            test_smiles=test_smiles,  # All test samples (num_test_samples)
                            epoch=epoch,
                            training_bytes=current_cumulative,
                            bytes_threshold=current_threshold,
                            output_dir=predictions_dir,
                            device=device,
                        )
                        print(f"[PREDICTIONS] ✓ Saved to {predictions_dir}/predictions_bytes_{current_threshold:,}.pkl.gz")
                    except Exception as e:
                        print(f"[PREDICTIONS] ✗ Failed to save: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[PREDICTIONS] Skipped (skip_visualization=True)")
                
                # Only save model checkpoint if loss has improved (or if this is the first checkpoint)
                should_save_checkpoint = (best_loss is None) or (current_loss < best_loss)
                
                if should_save_checkpoint:
                    # Update best loss
                    best_loss = current_loss
                    # Save byte-based checkpoint (only when loss improves)
                    try:
                        save_checkpoint('bytes', epoch=epoch, bytes_threshold=current_threshold, metrics=current_metrics, cumulative_bytes=current_cumulative)
                        print(f"[CHECKPOINT] ✓ Saved model checkpoint at {current_threshold:,} bytes (loss improved: {current_loss:.4f})")
                    except Exception as e:
                        print(f"[CHECKPOINT] ✗ Failed to save checkpoint: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue training even if checkpoint save fails
                else:
                    print(f"[CHECKPOINT] Skipped model checkpoint (loss not improved: {current_loss:.4f} >= {best_loss:.4f})")
                
                last_checkpoint_bytes = current_threshold
                
                # Always update metadata with stats (for visualize_stats.py) regardless of loss improvement
                metadata['training_history'].append({
                    'checkpoint_type': 'bytes',
                    'bytes_threshold': current_threshold,
                    'cumulative_training_bytes': current_cumulative,
                    'metrics': current_metrics,
                })
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        # Reset epoch metrics accumulator
        epoch_metrics_accumulator = {
            'total_loss': 0.0,
            'total_ce_loss': 0.0,
            'total_lb_loss': 0.0,
            'num_batches': 0,
        }
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train with checkpoint callback
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            args.gradient_accumulation,
            checkpoint_callback=checkpoint_callback,
            metrics_accumulator=epoch_metrics_accumulator,
        )
        
        # Update cumulative training bytes
        epoch_bytes = train_metrics.get('training_bytes', 0)
        cumulative_training_bytes += epoch_bytes
        
        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"(CE: {train_metrics['ce_loss']:.4f}, "
              f"LB: {train_metrics['lb_loss']:.6f})")
        print(f"Training bytes this epoch: {epoch_bytes:,}, Cumulative: {cumulative_training_bytes:,}")
        
        # Save epoch checkpoint
        save_checkpoint('epoch', epoch=epoch, metrics=train_metrics)
        
        # Update metadata
        metadata['training_history'].append({
            'epoch': epoch + 1,
            'checkpoint_type': 'epoch',
            'metrics': train_metrics,
            'cumulative_training_bytes': cumulative_training_bytes,
            'training_bytes_this_epoch': epoch_bytes,
        })
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Check for shutdown request
        if shutdown_requested:
            print("\nShutdown requested. Saving final checkpoint...")
            save_checkpoint('final', epoch=epoch, metrics=train_metrics)
            
            # Run visualization even if training was stopped
            if not args.skip_visualization:
                # Run visualization directly using boundary_predictions.json
                print("\nRunning visualization from boundary_predictions.json...")
                try:
                    import subprocess
                    import sys
                    
                    visualize_script = Path(__file__).parent / "visualizations" / "visualize_training_evolution.py"
                    output_path = visualizations_dir / "training_evolution.gif"
                    
                    cmd = [
                        sys.executable,
                        str(visualize_script),
                        "--run-dir", str(run_dir),
                        "--output", str(output_path),
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"\n✓ Visualization complete! Saved to {output_path}")
                    else:
                        print(f"\n✗ Visualization failed:")
                        print(result.stderr)
                except Exception as e:
                    print(f"\n✗ Visualization failed: {e}")
                    import traceback
                    traceback.print_exc()
            break
    
    print(f"\nTraining complete!")
    print(f"Run directory: {run_dir}")
    print(f"Total training bytes: {cumulative_training_bytes:,}")
    print(f"Checkpoints saved in: {checkpoints_dir}")
    print(f"Metadata saved in: {metadata_path}")
    
    # Run visualization on test SMILES if not skipped
    if not args.skip_visualization:
        # Run visualization directly using boundary_predictions.json
        print("\nRunning visualization from boundary_predictions.json...")
        try:
            import subprocess
            import sys
            
            visualize_script = Path(__file__).parent / "visualizations" / "visualize_training_evolution.py"
            output_path = visualizations_dir / "training_evolution.gif"
            
            cmd = [
                sys.executable,
                str(visualize_script),
                "--run-dir", str(run_dir),
                "--output", str(output_path),
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"\n✓ Visualization complete! Saved to {output_path}")
            else:
                print(f"\n✗ Visualization failed:")
                print(result.stderr)
        except Exception as e:
            print(f"\n✗ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping visualization (--skip-visualization flag set).")


if __name__ == '__main__':
    main()