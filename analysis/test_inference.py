#!/usr/bin/env python3
"""
Test script to validate H-Net tokenization inference on a small subset of data.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.utils.inference import load_model, extract_tokenization
from hnet.utils.tokenizers import ByteTokenizer


def test_single_smiles():
    """Test on a single SMILES string."""
    print("=" * 80)
    print("TEST 1: Single SMILES String")
    print("=" * 80)
    
    # Test checkpoint - using the new 1-epoch concatenated model
    checkpoint_dir = project_root / 'checkpoints' / 'run_large_20251113_181705'
    
    print(f"\nLoading model from: {checkpoint_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model, info = load_model(str(checkpoint_dir), device=device)
    
    print(f"\nModel info:")
    print(f"  Run name: {info['run_name']}")
    print(f"  Dataset: {info['dataset']}")
    print(f"  Dataset type: {info['dataset_type']}")
    print(f"  Concatenate: {info['concatenate']}")
    print(f"  Num concatenate: {info['num_concatenate']}")
    
    # Test SMILES
    test_smiles = "CC(C)CCC1C2CCC3C(C2CC1)CCC3C(C)CCCC(C)C"
    print(f"\nTest SMILES: {test_smiles}")
    print(f"Length: {len(test_smiles)} characters")
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Run tokenization
    print("\nRunning tokenization...")
    result = extract_tokenization(model, test_smiles, tokenizer, device)
    
    # Display results
    print(f"\nResults:")
    print(f"  Number of tokens: {result['num_tokens']}")
    print(f"  Tokens: {result['tokens']}")
    print(f"  Breakpoints: {result['breakpoints']}")
    print(f"  Breakpoint chars: {result['breakpoint_chars']}")
    
    # Validate that tokens reconstruct the original text
    reconstructed = ''.join(result['tokens'])
    if reconstructed == test_smiles:
        print(f"\n✓ Token reconstruction matches original text!")
    else:
        print(f"\n✗ ERROR: Token reconstruction doesn't match!")
        print(f"  Original:      {test_smiles}")
        print(f"  Reconstructed: {reconstructed}")
    
    return result


def test_multiple_smiles():
    """Test on multiple SMILES strings from dataset."""
    print("\n" + "=" * 80)
    print("TEST 2: Multiple SMILES from Dataset (10 samples)")
    print("=" * 80)
    
    # Load model
    checkpoint_dir = project_root / 'checkpoints' / 'run_large_20251113_181705'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading model from: {checkpoint_dir}")
    model, info = load_model(str(checkpoint_dir), device=device)
    
    # Load dataset
    import pandas as pd
    dataset_csv = info['dataset_csv']
    print(f"\nLoading dataset from: {dataset_csv}")
    
    df = pd.read_csv(dataset_csv)
    smiles_col = 'SMILES' if info['dataset_type'] == 'PI1M' else 'smiles'
    smiles_list = df[smiles_col].dropna().head(10).tolist()
    
    print(f"Testing on {len(smiles_list)} SMILES strings...")
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Process each SMILES
    results = []
    for i, smiles in enumerate(smiles_list):
        print(f"\n[{i+1}/{len(smiles_list)}] Processing: {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
        
        try:
            result = extract_tokenization(model, smiles, tokenizer, device)
            results.append(result)
            
            # Validate reconstruction
            reconstructed = ''.join(result['tokens'])
            if reconstructed == smiles:
                print(f"  ✓ {result['num_tokens']} tokens, reconstruction OK")
            else:
                print(f"  ✗ ERROR: Reconstruction mismatch!")
                print(f"    Original:      {smiles}")
                print(f"    Reconstructed: {reconstructed}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if results:
        num_tokens_list = [r['num_tokens'] for r in results]
        avg_tokens = sum(num_tokens_list) / len(num_tokens_list)
        min_tokens = min(num_tokens_list)
        max_tokens = max(num_tokens_list)
        
        print(f"Successfully processed: {len(results)}/{len(smiles_list)} samples")
        print(f"Average tokens per SMILES: {avg_tokens:.2f}")
        print(f"Min tokens: {min_tokens}")
        print(f"Max tokens: {max_tokens}")
        
        # Calculate average token length
        all_token_lengths = []
        for r in results:
            all_token_lengths.extend([len(t) for t in r['tokens']])
        
        if all_token_lengths:
            avg_token_len = sum(all_token_lengths) / len(all_token_lengths)
            print(f"Average token length: {avg_token_len:.2f} characters")
    else:
        print("No successful results!")
    
    return results


def test_concatenated_smiles():
    """Test on concatenated SMILES (if applicable)."""
    print("\n" + "=" * 80)
    print("TEST 3: Concatenated SMILES")
    print("=" * 80)
    
    # Load model
    checkpoint_dir = project_root / 'checkpoints' / 'run_large_20251113_181705'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading model from: {checkpoint_dir}")
    model, info = load_model(str(checkpoint_dir), device=device)
    
    # Check if model was trained with concatenation
    if not info['concatenate']:
        print("\nModel was not trained with concatenation. Skipping this test.")
        return None
    
    print(f"Model was trained with {info['num_concatenate']}-SMILES concatenation")
    
    # Load dataset
    import pandas as pd
    dataset_csv = info['dataset_csv']
    df = pd.read_csv(dataset_csv)
    smiles_col = 'SMILES' if info['dataset_type'] == 'PI1M' else 'smiles'
    smiles_list = df[smiles_col].dropna().head(info['num_concatenate']).tolist()
    
    # Create concatenated SMILES (using space separator as in training)
    concatenated = ' '.join(smiles_list)
    
    print(f"\nConcatenated SMILES ({len(smiles_list)} molecules):")
    print(f"  Length: {len(concatenated)} characters")
    print(f"  Preview: {concatenated[:100]}...")
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Run tokenization
    print("\nRunning tokenization...")
    result = extract_tokenization(model, concatenated, tokenizer, device)
    
    # Display results
    print(f"\nResults:")
    print(f"  Number of tokens: {result['num_tokens']}")
    print(f"  Average token length: {len(concatenated) / result['num_tokens']:.2f} characters")
    
    # Show first 10 tokens
    print(f"\nFirst 10 tokens:")
    for i, token in enumerate(result['tokens'][:10]):
        print(f"    [{i}] '{token}'")
    
    # Validate reconstruction
    reconstructed = ''.join(result['tokens'])
    if reconstructed == concatenated:
        print(f"\n✓ Token reconstruction matches original text!")
    else:
        print(f"\n✗ ERROR: Token reconstruction doesn't match!")
    
    return result


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("H-NET TOKENIZATION INFERENCE TEST SUITE")
    print("=" * 80)
    
    try:
        # Test 1: Single SMILES
        result1 = test_single_smiles()
        
        # Test 2: Multiple SMILES
        result2 = test_multiple_smiles()
        
        # Test 3: Concatenated SMILES
        result3 = test_concatenated_smiles()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED!")
        print("=" * 80)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n✓ GPU memory cleared")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

