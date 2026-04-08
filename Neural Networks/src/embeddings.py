"""
Embeddings Module for Transformer Architecture

Implements:
- Token Embeddings: Convert discrete tokens to continuous vectors
- Positional Encodings: Add position information using sinusoidal functions

Mathematical formulation for positional encoding:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer.
    
    Converts discrete token IDs to continuous vector representations.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimensionality of embeddings
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for token embedding.
        
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
        
        Returns:
            Embedded tokens of shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) as in the original paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sinusoidal functions.
    
    Since the Transformer has no recurrence or convolution, we inject
    positional information using sine and cosine functions of different frequencies.
    
    Mathematical Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    This allows the model to easily learn to attend by relative positions,
    as for any fixed offset k, PE(pos+k) can be represented as a linear
    function of PE(pos).
    
    Args:
        d_model: Dimensionality of the model
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term for the sinusoidal functions
        # div_term = 1 / (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_seq_length, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of module state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
        
        Returns:
            Embeddings with positional encoding (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # Add positional encoding (broadcasting over batch dimension)
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Combined Token Embedding and Positional Encoding.
    
    This module combines token embeddings with positional encodings
    to create the final input embeddings for the Transformer.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimensionality of embeddings
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super(TransformerEmbedding, self).__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings with positional encoding.
        
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
        
        Returns:
            Final embeddings of shape (batch_size, seq_len, d_model)
        """
        # Get token embeddings
        token_emb = self.token_embedding(x)
        
        # Add positional encoding
        output = self.positional_encoding(token_emb)
        
        return output


if __name__ == "__main__":
    # Demo: Visualize positional encodings
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("Testing Transformer Embeddings\n")
    
    vocab_size = 10000
    d_model = 512
    max_seq_length = 100
    batch_size = 2
    seq_len = 50
    
    # Create embedding module
    embedding = TransformerEmbedding(vocab_size, d_model, max_seq_length)
    
    # Create sample input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = embedding(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in embedding.parameters()):,}")
    
    # Visualize positional encodings
    print("\n" + "="*50)
    print("Visualizing Positional Encodings\n")
    
    pe_module = PositionalEncoding(d_model, max_seq_length)
    pos_encoding = pe_module.pe.squeeze(0).numpy()
    
    print(f"Positional encoding shape: {pos_encoding.shape}")
    print(f"Position encoding for position 0 (first 10 dims): {pos_encoding[0, :10]}")
    print(f"Position encoding for position 10 (first 10 dims): {pos_encoding[10, :10]}")
    
    # Optional: Create visualization if matplotlib is available
    try:
        plt.figure(figsize=(12, 6))
        plt.imshow(pos_encoding[:50, :].T, cmap='RdBu', aspect='auto')
        plt.colorbar()
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.title('Positional Encoding Visualization (first 50 positions)')
        plt.tight_layout()
        print("\n(Positional encoding heatmap would be displayed here)")
        # plt.savefig('positional_encoding.png')
        # plt.show()
    except ImportError:
        print("Note: Install matplotlib to visualize positional encodings")
    
    # Show how positional encodings vary across dimensions
    print("\n" + "="*50)
    print("Positional Encoding Statistics:\n")
    
    for pos in [0, 10, 25, 50]:
        pe_pos = pos_encoding[pos, :]
        print(f"Position {pos:3d}: mean={pe_pos.mean():.4f}, std={pe_pos.std():.4f}, "
              f"min={pe_pos.min():.4f}, max={pe_pos.max():.4f}")
