#!/usr/bin/env python3
"""
Generate SMILES strings from trained HNet model.
Adapted from the original generate.py for SMILES generation.
"""

import torch
import json
import argparse
import sys
from pathlib import Path
from omegaconf import ListConfig

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.utils.tokenizers import ByteTokenizer


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_from_checkpoint(checkpoint_path: str, config_path: str):
    """Load model from checkpoint."""
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create config objects
    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    # Create model
    device = get_device()
    dtype = torch.bfloat16 if device.type != 'cpu' else torch.float32
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
    model.eval()

    # Load checkpoint
    major, minor = map(int, torch.__version__.split('.')[:2])
    if (major, minor) >= (2, 6):
        with torch.serialization.safe_globals([ListConfig]):
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    return model


def generate(
    model,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.9,
):
    """Generate SMILES from the model, yielding tokens as they're generated."""
    device = next(model.parameters()).device
    tokenizer = ByteTokenizer()

    # Tokenize prompt
    encoded = tokenizer.encode([prompt], add_bos=True)[0]
    input_ids = torch.tensor(
        encoded["input_ids"], dtype=torch.long, device=device
    ).unsqueeze(0)

    inference_cache = model.allocate_inference_cache(
        1, input_ids.shape[1] + max_tokens, dtype=torch.bfloat16 if device.type != 'cpu' else torch.float32
    )

    with torch.inference_mode():
        mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        output = model.forward(input_ids, mask=mask, inference_params=inference_cache)

    logits = output.logits[0, -1, :] / temperature

    for _ in range(max_tokens):
        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        if next_token.item() == tokenizer.eos_idx:
            break

        current_token = next_token.unsqueeze(0)
        yield current_token

        with torch.inference_mode():
            output = model.step(current_token, inference_cache)

        # Get logits and apply temperature
        logits = output.logits[0, -1, :] / temperature


def main():
    parser = argparse.ArgumentParser(description="Generate SMILES from an H-Net model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model configuration (.json file)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="*",
        help="Starting prompt (default: '*')",
    )
    args = parser.parse_args()

    print("Loading model...")
    try:
        model = load_from_checkpoint(args.checkpoint, args.config)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    tokenizer = ByteTokenizer()

    prompt = args.prompt

    print(
        f"\nGenerating (max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})"
    )

    print(f"\033[92m{prompt}\033[0m", end="")
    token_count = 0
    buf = []

    for token in generate(
        model,
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ):
        buf.append(token)
        token_count += 1

        decoded = None
        res = None
        for j in range(1, min(len(buf), 4)):
            try:
                res = tokenizer.decode(buf[:j])
                decoded = j
            except:
                pass

        if res is not None:
            print(res, end="", flush=True)
            buf = buf[decoded:]

    print()  # Newline at end


if __name__ == "__main__":
    main()

