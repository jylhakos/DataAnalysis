"""
Encoder Module for Transformer Architecture

Implements the encoder stack which consists of N identical layers.
Each layer has two sub-layers:
1. Multi-head self-attention mechanism
2. Position-wise feed-forward network

Each sub-layer has a residual connection and layer normalization.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer.
    
    Consists of:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network
    
    Both sub-layers use:
    - Residual connections: output = LayerNorm(x + Sublayer(x))
    - Layer normalization
    
    Args:
        d_model: Dimensionality of the model
        n_heads: Number of attention heads
        d_ff: Dimensionality of feed-forward network
        dropout: Dropout probability
        activation: Activation function for feed-forward network
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout, activation)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for encoder layer.
        
        Mathematical operations:
        1. x = LayerNorm(x + MultiHeadAttention(x, x, x))
        2. x = LayerNorm(x + FeedForward(x))
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor for padding
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights from self-attention
        """
        # Multi-head self-attention with residual connection and layer norm
        # Note: In self-attention, query = key = value = x
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attention_weights


class Encoder(nn.Module):
    """
    Transformer Encoder.
    
    Stack of N identical encoder layers. Each layer processes the input
    sequentially, with the output of one layer feeding into the next.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimensionality of the model
        n_layers: Number of encoder layers
        n_heads: Number of attention heads
        d_ff: Dimensionality of feed-forward network
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
        activation: Activation function for feed-forward networks
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super(Encoder, self).__init__()
        
        # Stack of N encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        self.n_layers = n_layers
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, list]:
        """
        Forward pass through all encoder layers.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            mask: Optional padding mask
        
        Returns:
            output: Encoded representations of shape (batch_size, seq_len, d_model)
            attention_weights_list: List of attention weights from each layer
        """
        attention_weights_list = []
        
        # Pass through each encoder layer
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            attention_weights_list.append(attention_weights)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        return x, attention_weights_list


if __name__ == "__main__":
    print("Testing Transformer Encoder\n")
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_layers = 6
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    
    # Create sample input (embeddings)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create encoder
    encoder = Encoder(d_model, n_layers, n_heads, d_ff, dropout)
    
    # Forward pass
    output, attention_weights_list = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of encoder layers: {n_layers}")
    print(f"Number of attention weight tensors: {len(attention_weights_list)}")
    print(f"Attention weights shape (per layer): {attention_weights_list[0].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with padding mask
    print("\n" + "="*50)
    print("Testing with Padding Mask\n")
    
    from .attention import create_padding_mask
    
    # Create sample sequence with padding
    seq_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                            [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]])
    pad_mask = create_padding_mask(seq_ids)
    
    print(f"Sequence IDs:\n{seq_ids}")
    print(f"Padding mask shape: {pad_mask.shape}")
    
    # Forward pass with mask
    output_masked, _ = encoder(x, pad_mask)
    print(f"Output with mask shape: {output_masked.shape}")
    
    # Verify output consistency
    print("\nOutput statistics (with mask):")
    print(f"  Mean: {output_masked.mean():.4f}")
    print(f"  Std: {output_masked.std():.4f}")
    print(f"  Min: {output_masked.min():.4f}")
    print(f"  Max: {output_masked.max():.4f}")
