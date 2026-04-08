"""
Unit tests for attention mechanisms.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    create_padding_mask,
    create_causal_mask,
    create_decoder_mask
)


class TestScaledDotProductAttention:
    """Test scaled dot-product attention."""
    
    def test_attention_shape(self):
        """Test output shape of attention mechanism."""
        batch_size = 2
        n_heads = 8
        seq_len = 10
        d_k = 64
        
        attention = ScaledDotProductAttention()
        
        Q = torch.randn(batch_size, n_heads, seq_len, d_k)
        K = torch.randn(batch_size, n_heads, seq_len, d_k)
        V = torch.randn(batch_size, n_heads, seq_len, d_k)
        
        output, weights = attention(Q, K, V)
        
        assert output.shape == (batch_size, n_heads, seq_len, d_k)
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 for each query."""
        batch_size = 2
        n_heads = 4
        seq_len = 5
        d_k = 32
        
        attention = ScaledDotProductAttention(dropout=0.0)  # No dropout for exact test
        
        Q = torch.randn(batch_size, n_heads, seq_len, d_k)
        K = torch.randn(batch_size, n_heads, seq_len, d_k)
        V = torch.randn(batch_size, n_heads, seq_len, d_k)
        
        _, weights = attention(Q, K, V)
        
        # Sum along last dimension should be 1
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


class TestMultiHeadAttention:
    """Test multi-head attention."""
    
    def test_multi_head_attention_shape(self):
        """Test output shape of multi-head attention."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        n_heads = 8
        
        mha = MultiHeadAttention(d_model, n_heads)
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = mha(query, key, value)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_self_attention(self):
        """Test self-attention (Q=K=V)."""
        batch_size = 2
        seq_len = 8
        d_model = 256
        n_heads = 4
        
        mha = MultiHeadAttention(d_model, n_heads)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, _ = mha(x, x, x)
        
        assert output.shape == x.shape
    
    def test_cross_attention(self):
        """Test cross-attention (different Q and K,V)."""
        batch_size = 2
        tgt_len = 5
        src_len = 10
        d_model = 256
        n_heads = 4
        
        mha = MultiHeadAttention(d_model, n_heads)
        
        query = torch.randn(batch_size, tgt_len, d_model)
        key = torch.randn(batch_size, src_len, d_model)
        value = torch.randn(batch_size, src_len, d_model)
        
        output, attention_weights = mha(query, key, value)
        
        assert output.shape == (batch_size, tgt_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, tgt_len, src_len)


class TestMasks:
    """Test masking functions."""
    
    def test_padding_mask_shape(self):
        """Test padding mask shape."""
        batch_size = 2
        seq_len = 10
        
        seq = torch.randint(0, 100, (batch_size, seq_len))
        mask = create_padding_mask(seq, pad_token_id=0)
        
        assert mask.shape == (batch_size, 1, 1, seq_len)
    
    def test_padding_mask_detects_padding(self):
        """Test that padding mask correctly identifies padding tokens."""
        seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        mask = create_padding_mask(seq, pad_token_id=0)
        
        # Check that padding positions are masked (0)
        assert mask[0, 0, 0, 0] == 1  # First token (not padding)
        assert mask[0, 0, 0, 3] == 0  # Fourth token (padding)
        assert mask[1, 0, 0, 2] == 0  # Third token (padding in second sequence)
    
    def test_causal_mask_shape(self):
        """Test causal mask shape."""
        seq_len = 10
        device = torch.device('cpu')
        
        mask = create_causal_mask(seq_len, device)
        
        assert mask.shape == (1, 1, seq_len, seq_len)
    
    def test_causal_mask_structure(self):
        """Test that causal mask has proper lower triangular structure."""
        seq_len = 5
        device = torch.device('cpu')
        
        mask = create_causal_mask(seq_len, device)
        mask_2d = mask.squeeze()
        
        # Check lower triangular structure
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    assert mask_2d[i, j] == 1
                else:
                    assert mask_2d[i, j] == 0
    
    def test_decoder_mask_combines_both(self):
        """Test that decoder mask combines padding and causal masks."""
        seq = torch.tensor([[1, 2, 3, 0, 0]])
        mask = create_decoder_mask(seq, pad_token_id=0)
        
        # Should have causal structure but also mask padding
        assert mask.shape == (1, 1, 5, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
