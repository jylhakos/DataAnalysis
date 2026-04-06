"""
Transformer Architecture

Implementation of transformer blocks and complete transformer model
based on "Attention Is All You Need" (Vaswani et al., 2017) and modern LLM architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    Used in modern LLMs (LLaMA, Gemma, Qwen) instead of LayerNorm.
    Simpler and more efficient, removes mean-centering.
    
    RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            d_model: Model dimension
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    Two-layer MLP with activation in between.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'swiglu')
        """
        super().__init__()
        
        self.activation_name = activation
        
        if activation == 'swiglu':
            # SwiGLU: used in LLaMA, Gemma
            # SwiGLU(x) = (xW₁ ⊙ swish(xW₂))W₃
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_model, d_ff, bias=False)
            self.w3 = nn.Linear(d_ff, d_model, bias=False)
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'gelu':
                self.activation = nn.GELU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        if self.activation_name == 'swiglu':
            # SwiGLU activation
            gate = F.silu(self.w1(x))  # swish activation
            x = gate * self.w2(x)
            x = self.w3(x)
        else:
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear2(x)
        
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Transformer Block (Decoder-style for language modeling)
    
    Architecture:
        x = x + Attention(LayerNorm(x))       # Pre-norm + residual
        x = x + FFN(LayerNorm(x))             # Pre-norm + residual
    
    Pre-norm has better gradient flow for deep transformers.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, use_rmsnorm: bool = True,
                 activation: str = 'gelu'):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            use_rmsnorm: Use RMSNorm instead of LayerNorm
            activation: Activation function in FFN
        """
        super().__init__()
        
        from attention import CausalSelfAttention
        
        # Self-attention
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        
        # Normalization layers
        if use_rmsnorm:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        x = x + self.dropout(self.attention(self.norm1(x)))
        
        # Feed-forward with residual connection
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x


class GPTModel(nn.Module):
    """
    GPT-style Language Model
    
    Decoder-only transformer for autoregressive language modeling.
    Similar to GPT-2, GPT-3, LLaMA, etc.
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 d_ff: int, max_seq_len: int = 1024, dropout: float = 0.1,
                 use_rmsnorm: bool = True, activation: str = 'gelu'):
        """
        Initialize GPT model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_rmsnorm: Use RMSNorm instead of LayerNorm
            activation: Activation function
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embeddings (learned, GPT-2 style)
        # Modern LLMs use RoPE instead, which is built into attention
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_rmsnorm, activation)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        if use_rmsnorm:
            self.ln_f = RMSNorm(d_model)
        else:
            self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection (language modeling head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between token embeddings and output projection
        # This is common in LLMs to reduce parameters
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using a scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of GPT model.
        
        Args:
            x: Input token indices (batch_size, seq_len)
            targets: Target token indices for computing loss (optional)
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Position embeddings
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens
            )
        
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            idx: Starting token indices (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (optional)
            
        Returns:
            Generated token indices (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = self(idx)
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def demo_transformer():
    """Demonstrate transformer model."""
    print("=" * 70)
    print("Transformer Architecture Demo")
    print("=" * 70)
    
    # Model configuration (GPT-2 small style)
    config = {
        'vocab_size': 50257,
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'd_ff': 3072,  # 4 * d_model
        'max_seq_len': 1024,
        'dropout': 0.1,
        'use_rmsnorm': False,  # Use LayerNorm for GPT-2 compatibility
        'activation': 'gelu'
    }
    
    print("\nModel Configuration:")
    print("-" * 70)
    for key, value in config.items():
        print(f"{key:15s}: {value}")
    
    # Create model
    model = GPTModel(**config)
    
    # Count parameters
    n_params = model.count_parameters()
    print(f"\nTotal parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Create dummy input
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    print("\n" + "-" * 70)
    print("Forward Pass")
    print("-" * 70)
    
    logits, loss = model(x, targets)
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    print("\n" + "-" * 70)
    print("Text Generation")
    print("-" * 70)
    
    start_tokens = torch.randint(0, config['vocab_size'], (1, 10))
    print(f"Starting with {start_tokens.shape[1]} tokens")
    
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"Generated {generated.shape[1]} tokens total")
    
    # Analyze architecture
    print("\n" + "-" * 70)
    print("Architecture Analysis")
    print("-" * 70)
    
    # Count parameters by component
    embedding_params = sum(p.numel() for p in model.token_embedding.parameters())
    embedding_params += sum(p.numel() for p in model.position_embedding.parameters())
    
    transformer_params = sum(p.numel() for p in model.blocks.parameters())
    
    print(f"Embedding parameters: {embedding_params:,} ({embedding_params/n_params*100:.1f}%)")
    print(f"Transformer parameters: {transformer_params:,} ({transformer_params/n_params*100:.1f}%)")
    print(f"Parameters per layer: {transformer_params/config['n_layers']:,.0f}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_transformer()
