"""
Self-Attention Mechanisms for Transformer Architecture

Implements:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Masked Multi-Head Attention (for decoder)

Mathematical formulation from "Attention Is All You Need" (2017):
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Computes attention scores using queries, keys, and values.
    The attention output is a weighted sum of values, where weights
    are computed from queries and keys.
    
    Mathematical Formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        dropout: Dropout probability for attention weights
    """
    
    def __init__(self, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, n_heads, seq_len, d_k)
            key: Key tensor of shape (batch_size, n_heads, seq_len, d_k)
            value: Value tensor of shape (batch_size, n_heads, seq_len, d_v)
            mask: Optional mask tensor to prevent attention to certain positions
        
        Returns:
            output: Attention output (batch_size, n_heads, seq_len, d_v)
            attention_weights: Attention weights (batch_size, n_heads, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        # Shape: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (e.g., for padding or causal masking)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum of values
        # Shape: (batch_size, n_heads, seq_len, d_v)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    
    Mathematical Formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Args:
        d_model: Dimensionality of input embeddings
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_heads, d_k).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor of shape (batch_size, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back to original shape.
        
        Args:
            x: Input tensor of shape (batch_size, n_heads, seq_len, d_k)
        
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, n_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor
        
        Returns:
            output: Attention output (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)
        
        # Linear projections and split into multiple heads
        # Shape: (batch_size, n_heads, seq_len, d_k)
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        
        # Expand mask for multi-head attention if provided
        # Mask should be (batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len)
        # No need to unsqueeze if it's already 4D
        
        # Apply scaled dot-product attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        # Shape: (batch_size, seq_len, d_model)
        attn_output = self.combine_heads(attn_output)
        
        # Apply output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create padding mask to prevent attention to padding tokens.
    
    Args:
        seq: Input sequence tensor (batch_size, seq_len)
        pad_token_id: ID of padding token
    
    Returns:
        Mask tensor (batch_size, 1, 1, seq_len) with 1 for real tokens, 0 for padding
    """
    # Create mask: 1 for real tokens, 0 for padding
    mask = (seq != pad_token_id).float().unsqueeze(1).unsqueeze(2)
    return mask


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (look-ahead) mask for decoder self-attention.
    
    Prevents positions from attending to subsequent positions.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
    
    Returns:
        Mask tensor (1, 1, seq_len, seq_len) with lower triangular structure
    """
    # Create lower triangular matrix (1 where we can attend, 0 where we can't)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.float))
    return mask.unsqueeze(0).unsqueeze(0)


def create_decoder_mask(
    tgt_seq: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Create combined mask for decoder (padding + causal).
    
    Args:
        tgt_seq: Target sequence tensor (batch_size, seq_len)
        pad_token_id: ID of padding token
    
    Returns:
        Combined mask tensor
    """
    seq_len = tgt_seq.size(1)
    device = tgt_seq.device
    
    # Padding mask (batch_size, 1, 1, seq_len)
    padding_mask = create_padding_mask(tgt_seq, pad_token_id)
    
    # Causal mask (1, 1, seq_len, seq_len)
    causal_mask = create_causal_mask(seq_len, device)
    
    # Combine masks: element-wise multiplication (both must be 1 to attend)
    combined_mask = padding_mask * causal_mask
    
    return combined_mask


if __name__ == "__main__":
    # Demo: Test attention mechanisms
    print("Testing Multi-Head Attention Mechanism\n")
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # Create random input
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize multi-head attention
    mha = MultiHeadAttention(d_model, n_heads)
    
    # Forward pass
    output, attention_weights = mha(query, key, value)
    
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nNumber of parameters: {sum(p.numel() for p in mha.parameters()):,}")
    
    # Test masking
    print("\n" + "="*50)
    print("Testing Masking Mechanisms\n")
    
    seq = torch.tensor([[1, 2, 3, 4, 0, 0], [1, 2, 0, 0, 0, 0]])
    padding_mask = create_padding_mask(seq)
    print(f"Sequence:\n{seq}")
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Padding mask:\n{padding_mask.squeeze()}")
    
    causal_mask = create_causal_mask(6, seq.device)
    print(f"\nCausal mask shape: {causal_mask.shape}")
    print(f"Causal mask:\n{causal_mask.squeeze()}")
