"""
Decoder Module for Transformer Architecture

Implements the decoder stack which consists of N identical layers.
Each layer has three sub-layers:
1. Masked multi-head self-attention
2. Multi-head cross-attention over encoder output
3. Position-wise feed-forward network

Each sub-layer has a residual connection and layer normalization.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer.
    
    Consists of:
    1. Masked Multi-Head Self-Attention (prevents looking ahead)
    2. Multi-Head Cross-Attention (attends to encoder output)
    3. Position-wise Feed-Forward Network
    
    All sub-layers use residual connections and layer normalization.
    
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
        super(DecoderLayer, self).__init__()
        
        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Multi-head cross-attention (decoder attends to encoder)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout, activation)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for decoder layer.
        
        Args:
            x: Decoder input of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Mask for target sequence (causal + padding)
            src_mask: Mask for source sequence (padding)
        
        Returns:
            output: Output tensor of shape (batch_size, tgt_seq_len, d_model)
            self_attn_weights: Self-attention weights
            cross_attn_weights: Cross-attention weights
        """
        # 1. Masked self-attention with residual connection and layer norm
        self_attn_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_output)
        
        # 2. Cross-attention with encoder output
        # Query comes from decoder, Key and Value come from encoder
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + cross_attn_output)
        
        # 3. Feed-forward network with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    """
    Transformer Decoder.
    
    Stack of N identical decoder layers. Each layer processes the target
    sequence and attends to the encoder output.
    
    Args:
        d_model: Dimensionality of the model
        n_layers: Number of decoder layers
        n_heads: Number of attention heads
        d_ff: Dimensionality of feed-forward network
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
        super(Decoder, self).__init__()
        
        # Stack of N decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        self.n_layers = n_layers
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, list, list]:
        """
        Forward pass through all decoder layers.
        
        Args:
            x: Target embeddings of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Mask for target sequence
            src_mask: Mask for source sequence
        
        Returns:
            output: Decoded representations of shape (batch_size, tgt_seq_len, d_model)
            self_attn_weights_list: List of self-attention weights from each layer
            cross_attn_weights_list: List of cross-attention weights from each layer
        """
        self_attn_weights_list = []
        cross_attn_weights_list = []
        
        # Pass through each decoder layer
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, tgt_mask, src_mask
            )
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        return x, self_attn_weights_list, cross_attn_weights_list


if __name__ == "__main__":
    print("Testing Transformer Decoder\n")
    
    # Hyperparameters
    batch_size = 2
    src_seq_len = 15
    tgt_seq_len = 10
    d_model = 512
    n_layers = 6
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    
    # Create sample inputs
    tgt_embeddings = torch.randn(batch_size, tgt_seq_len, d_model)
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    
    # Create decoder
    decoder = Decoder(d_model, n_layers, n_heads, d_ff, dropout)
    
    # Forward pass
    output, self_attn_weights_list, cross_attn_weights_list = decoder(
        tgt_embeddings, encoder_output
    )
    
    print(f"Target embeddings shape: {tgt_embeddings.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder output shape: {output.shape}")
    print(f"Number of decoder layers: {n_layers}")
    print(f"Number of self-attention weight tensors: {len(self_attn_weights_list)}")
    print(f"Number of cross-attention weight tensors: {len(cross_attn_weights_list)}")
    print(f"Self-attention weights shape: {self_attn_weights_list[0].shape}")
    print(f"Cross-attention weights shape: {cross_attn_weights_list[0].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with masks
    print("\n" + "="*50)
    print("Testing with Masks\n")
    
    from .attention import create_padding_mask, create_decoder_mask
    
    # Create sample sequences
    src_seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 0],
                            [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0]])
    tgt_seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0],
                            [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]])
    
    src_mask = create_padding_mask(src_seq)
    tgt_mask = create_decoder_mask(tgt_seq)
    
    print(f"Source sequence shape: {src_seq.shape}")
    print(f"Target sequence shape: {tgt_seq.shape}")
    print(f"Source mask shape: {src_mask.shape}")
    print(f"Target mask shape: {tgt_mask.shape}")
    
    # Forward pass with masks
    output_masked, _, _ = decoder(tgt_embeddings, encoder_output, tgt_mask, src_mask)
    print(f"\nOutput with masks shape: {output_masked.shape}")
    
    # Verify output statistics
    print("\nOutput statistics (with masks):")
    print(f"  Mean: {output_masked.mean():.4f}")
    print(f"  Std: {output_masked.std():.4f}")
    print(f"  Min: {output_masked.min():.4f}")
    print(f"  Max: {output_masked.max():.4f}")
    
    # Demonstrate attention weight shapes
    print("\n" + "="*50)
    print("Attention Weight Shapes:\n")
    print("Self-attention (targets attend to targets):")
    print(f"  Shape: {self_attn_weights_list[0].shape}")
    print(f"  Interpretation: (batch_size={batch_size}, n_heads={n_heads}, "
          f"tgt_len={tgt_seq_len}, tgt_len={tgt_seq_len})")
    
    print("\nCross-attention (targets attend to source):")
    print(f"  Shape: {cross_attn_weights_list[0].shape}")
    print(f"  Interpretation: (batch_size={batch_size}, n_heads={n_heads}, "
          f"tgt_len={tgt_seq_len}, src_len={src_seq_len})")
