"""
Simple Training Example

This script demonstrates basic usage of the Transformer model for a copy task.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.transformer import Transformer
from src.config import TransformerConfig


def main():
    """Run a simple training example."""
    print("="*60)
    print("Simple Transformer Training Example")
    print("="*60)
    
    # Create small configuration
    config = TransformerConfig.transformer_small()
    
    print("\nModel Configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  d_ff: {config.d_ff}")
    
    # Create model
    print("\nCreating model...")
    model = Transformer(config)
    
    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Encoder parameters: {param_counts['encoder']:,}")
    print(f"  Decoder parameters: {param_counts['decoder']:,}")
    print(f"  Embedding parameters: {param_counts['embeddings']:,}")
    
    # Create sample data
    print("\nCreating sample data...")
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 8
    
    # Random token IDs (excluding special tokens 0, 1, 2)
    src = torch.randint(3, config.vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(3, config.vocab_size, (batch_size, tgt_seq_len))
    
    print(f"  Source shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        output, attention_weights = model(src, tgt)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (batch_size={batch_size}, tgt_seq_len={tgt_seq_len}, vocab_size={config.vocab_size})")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(src[:1], max_length=15, temperature=1.0)
    print(f"  Generated sequence shape: {generated.shape}")
    print(f"  Generated tokens: {generated[0].tolist()}")
    
    # Test different sampling strategies
    print("\nTesting sampling strategies:")
    
    print("  1. Greedy (temperature=0.1):")
    greedy = model.generate(src[:1], max_length=10, temperature=0.1)
    print(f"     {greedy[0].tolist()}")
    
    print("  2. Top-k sampling (k=50):")
    topk = model.generate(src[:1], max_length=10, temperature=1.0, top_k=50)
    print(f"     {topk[0].tolist()}")
    
    print("  3. Nucleus sampling (p=0.9):")
    nucleus = model.generate(src[:1], max_length=10, temperature=1.0, top_p=0.9)
    print(f"     {nucleus[0].tolist()}")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
    
    print("\nNext steps:")
    print("  1. Train the model: python train.py --epochs 10 --batch-size 32")
    print("  2. Run inference: python inference.py --model-path checkpoints/best_model.pt --interactive")
    print("  3. Start server: python server.py --model-path checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
