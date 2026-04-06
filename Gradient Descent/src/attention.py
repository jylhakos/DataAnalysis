"""
Attention Mechanisms for Transformers

Implementation of attention mechanisms from "Attention Is All You Need" (Vaswani et al., 2017)
and modern variants used in LLMs.

Mathematical foundations:
- Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Multi-Head Attention: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    The scaling factor 1/√d_k prevents the dot products from growing too large,
    which would push the softmax into regions with extremely small gradients.
    """
    
    def __init__(self, dropout: float = 0.1):
        """
        Initialize scaled dot-product attention.
        
        Args:
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor (batch_size, n_heads, seq_len, d_k)
            key: Key tensor (batch_size, n_heads, seq_len, d_k)
            value: Value tensor (batch_size, n_heads, seq_len, d_v)
            mask: Attention mask (optional)
            
        Returns:
            output: Attention output (batch_size, n_heads, seq_len, d_v)
            attention_weights: Attention scores (batch_size, n_heads, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum of values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Attention mask (optional)
            
        Returns:
            output: Attention output (batch_size, seq_len, d_model)
            attention_weights: Attention scores
        """
        batch_size = query.size(0)
        
        # Linear projections and split into multiple heads
        # Shape: (batch_size, seq_len, d_model) → (batch_size, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        # Shape: (batch_size, n_heads, seq_len, d_k) → (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)
    
    Used in modern LLMs like LLaMA 2/3, Gemma, Qwen.
    Reduces memory and computation compared to Multi-Head Attention by sharing
    key and value projections across multiple query heads.
    
    - MHA: Each head has separate Q, K, V
    - GQA: Multiple query heads share the same K, V
    - MQA: All query heads share a single K, V
    """
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        """
        Initialize grouped-query attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of query heads
            n_kv_heads: Number of key-value heads (< n_heads)
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # How many Q heads per KV head
        self.d_k = d_model // n_heads
        
        # Query projections (full number of heads)
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        
        # Key and value projections (reduced number of heads)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of grouped-query attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Attention mask (optional)
            
        Returns:
            output: Attention output (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        # Repeat K and V to match number of Q heads
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.d_k
        )
        
        output = self.W_o(attn_output)
        
        return output


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    Used in modern LLMs (LLaMA, Qwen, Gemma, etc.)
    Encodes position information through rotation in complex space.
    
    Better extrapolation to longer sequences than learned positional embeddings.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            d_model: Model dimension (must be even)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute rotation matrix for max_seq_len
        self._precompute_freqs(max_seq_len)
    
    def _precompute_freqs(self, seq_len: int):
        """Precompute rotation frequencies."""
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer('freqs_cos', freqs.cos())
        self.register_buffer('freqs_sin', freqs.sin())
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimensions."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary position embeddings.
        
        Args:
            x: Input tensor (batch_size, n_heads, seq_len, d_k)
            seq_len: Sequence length
            
        Returns:
            Tensor with position information encoded
        """
        # Ensure we have precomputed frequencies for this sequence length
        if seq_len > self.max_seq_len:
            self._precompute_freqs(seq_len)
        
        cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        return (x * cos) + (self.rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention for language modeling.
    
    Prevents positions from attending to future positions (autoregressive).
    Used in GPT-style decoder-only models.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_rope: bool = True, max_seq_len: int = 2048):
        """
        Initialize causal self-attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            use_rope: Whether to use RoPE
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_rope = use_rope
        
        # QKV projection
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE (optional)
        if use_rope:
            self.rope = RoPE(self.d_k, max_seq_len)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (will be created dynamically)
        self.register_buffer('mask', None)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask to prevent attending to future positions."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(
            1, 1, seq_len, seq_len
        )
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of causal self-attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            output: Attention output (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.W_qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        
        # Reshape to (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            Q = self.rope(Q, seq_len)
            K = self.rope(K, seq_len)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        if self.mask is None or self.mask.size(-1) != seq_len:
            self.mask = self._create_causal_mask(seq_len, x.device)
        
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.W_o(attn_output)
        
        return output


def demo_attention_mechanisms():
    """Demonstrate attention mechanisms."""
    print("=" * 70)
    print("Attention Mechanisms Demo")
    print("=" * 70)
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    print(f"d_model: {d_model}, n_heads: {n_heads}, seq_len: {seq_len}")
    
    # Test Multi-Head Attention
    print("\n" + "-" * 70)
    print("1. Multi-Head Attention")
    mha = MultiHeadAttention(d_model, n_heads)
    output, attn_weights = mha(x, x, x)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test Grouped-Query Attention
    print("\n" + "-" * 70)
    print("2. Grouped-Query Attention (GQA)")
    n_kv_heads = 2  # LLaMA-style: 8 query heads, 2 KV heads
    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    output_gqa = gqa(x, x, x)
    print(f"Output shape: {output_gqa.shape}")
    print(f"KV heads: {n_kv_heads}, Query heads: {n_heads}, Ratio: {n_heads//n_kv_heads}:1")
    
    # Test Causal Self-Attention with RoPE
    print("\n" + "-" * 70)
    print("3. Causal Self-Attention with RoPE")
    causal_attn = CausalSelfAttention(d_model, n_heads, use_rope=True)
    output_causal = causal_attn(x)
    print(f"Output shape: {output_causal.shape}")
    print("Position encoding: RoPE (Rotary Position Embeddings)")
    
    # Count parameters
    print("\n" + "-" * 70)
    print("4. Parameter Comparison")
    mha_params = sum(p.numel() for p in mha.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())
    causal_params = sum(p.numel() for p in causal_attn.parameters())
    
    print(f"Multi-Head Attention: {mha_params:,} parameters")
    print(f"Grouped-Query Attention: {gqa_params:,} parameters")
    print(f"Causal Self-Attention: {causal_params:,} parameters")
    print(f"GQA parameter reduction: {(1 - gqa_params/mha_params)*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_attention_mechanisms()
