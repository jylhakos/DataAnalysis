"""
Attention Mechanism Demonstration

This script demonstrates how attention heads work in transformers,
showing how information is transferred between word vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleAttention(nn.Module):
    """
    A simple implementation of the attention mechanism.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    
    def __init__(self, embed_dim):
        super(SimpleAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        """
        Forward pass through attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor
            
        Returns:
            output: Attention output
            attention_weights: Attention weight matrix
        """
        # Generate Q, K, V matrices
        Q = self.W_q(x)  # (batch_size, seq_len, embed_dim)
        K = self.W_k(x)  # (batch_size, seq_len, embed_dim)
        V = self.W_v(x)  # (batch_size, seq_len, embed_dim)
        
        # Calculate attention scores
        # scores = Q * K^T / sqrt(d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation.
    
    Allows the model to attend to information from different representation
    subspaces at different positions.
    """
    
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
    def split_heads(self, x):
        """Split the last dimension into (num_heads, head_dim)."""
        batch_size, seq_len, embed_dim = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self, x, mask=None):
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor
            
        Returns:
            output: Multi-head attention output
            attention_weights: Attention weights from all heads
        """
        batch_size = x.size(0)
        
        # Generate Q, K, V
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.embed_dim)
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights


def demonstrate_attention():
    """
    Demonstrate how attention mechanism works with a simple example.
    """
    print("=" * 60)
    print("ATTENTION MECHANISM DEMONSTRATION")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters
    embed_dim = 64
    seq_len = 5
    batch_size = 1
    
    # Create simple attention layer
    attention = SimpleAttention(embed_dim)
    
    # Create sample input (representing word embeddings)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"\nInput shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {embed_dim}")
    
    # Forward pass
    output, attention_weights = attention(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Display attention weights
    print("\nAttention Weight Matrix:")
    print("(Shows how much each position attends to other positions)")
    print("-" * 60)
    
    weights = attention_weights[0].detach().numpy()
    
    # Print header
    print("     ", end="")
    for j in range(seq_len):
        print(f"Pos{j}  ", end="")
    print()
    
    # Print weights
    for i in range(seq_len):
        print(f"Pos{i} ", end="")
        for j in range(seq_len):
            print(f"{weights[i][j]:.3f} ", end="")
        print()
    
    print("\nInterpretation:")
    print("- Each row shows how much that position attends to all positions")
    print("- Higher values indicate stronger attention")
    print("- Row sums equal 1.0 (due to softmax normalization)")


def demonstrate_multi_head_attention():
    """
    Demonstrate multi-head attention mechanism.
    """
    print("\n" + "=" * 60)
    print("MULTI-HEAD ATTENTION DEMONSTRATION")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    
    # Parameters
    embed_dim = 64
    num_heads = 8
    seq_len = 5
    batch_size = 1
    
    # Create multi-head attention
    mha = MultiHeadAttention(embed_dim, num_heads)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"\nInput shape: {x.shape}")
    print(f"  Number of attention heads: {num_heads}")
    print(f"  Dimension per head: {embed_dim // num_heads}")
    
    # Forward pass
    output, attention_weights = mha(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  (batch_size, num_heads, seq_len, seq_len)")
    
    print("\nAttention patterns by head:")
    print("-" * 60)
    
    # Show attention from first head
    first_head = attention_weights[0, 0].detach().numpy()
    
    print(f"\nHead 0 attention weights:")
    print("     ", end="")
    for j in range(seq_len):
        print(f"Pos{j}  ", end="")
    print()
    
    for i in range(seq_len):
        print(f"Pos{i} ", end="")
        for j in range(seq_len):
            print(f"{first_head[i][j]:.3f} ", end="")
        print()
    
    print("\nKey Insight:")
    print("- Different heads learn to attend to different patterns")
    print("- This allows the model to capture multiple types of relationships")
    print("- Heads can focus on syntax, semantics, or other linguistic features")


def simulate_word_attention():
    """
    Simulate attention on actual words to show practical application.
    """
    print("\n" + "=" * 60)
    print("WORD-LEVEL ATTENTION SIMULATION")
    print("=" * 60)
    
    # Example sentence
    sentence = ["The", "cat", "sat", "on", "mat"]
    
    print(f"\nSentence: {' '.join(sentence)}")
    
    # Simulate attention from "sat" to other words
    # In practice, these would be learned by the model
    attention_from_sat = {
        "The": 0.05,
        "cat": 0.40,  # High attention to subject
        "sat": 0.20,  # Self-attention
        "on": 0.15,
        "mat": 0.20   # Attention to location
    }
    
    print("\nWhen processing 'sat', attention distribution:")
    print("-" * 40)
    for word, weight in attention_from_sat.items():
        bar = "█" * int(weight * 50)
        print(f"{word:8s} {weight:.2f} {bar}")
    
    print("\nInterpretation:")
    print("- 'sat' pays most attention to 'cat' (the subject)")
    print("- Also attends to 'mat' (where the sitting happens)")
    print("- Some self-attention to maintain context")
    print("- This helps the model understand sentence structure")


if __name__ == "__main__":
    print("\n" + "*" * 60)
    print("TRANSFORMER ATTENTION MECHANISMS")
    print("Demonstrating how attention heads transfer information")
    print("*" * 60)
    
    # Run demonstrations
    demonstrate_attention()
    demonstrate_multi_head_attention()
    simulate_word_attention()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Attention allows models to focus on relevant inputs")
    print("2. Multi-head attention captures different types of relationships")
    print("3. Attention weights show which words influence others")
    print("4. This mechanism is fundamental to transformer models")
    print()
