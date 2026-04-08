"""
Configuration file for Transformer model hyperparameters.

This module defines the TransformerConfig class which holds all
hyperparameters needed to instantiate a Transformer model.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TransformerConfig:
    """
    Configuration class for Transformer model.
    
    Following the architecture from "Attention Is All You Need" (Vaswani et al., 2017),
    this configuration defines all hyperparameters for the model.
    
    Attributes:
        vocab_size (int): Size of the vocabulary
        d_model (int): Dimensionality of embeddings and hidden states
        n_heads (int): Number of attention heads in multi-head attention
        n_layers (int): Number of encoder and decoder layers
        d_ff (int): Dimensionality of feed-forward network
        max_seq_length (int): Maximum sequence length
        dropout (float): Dropout probability
        learning_rate (float): Initial learning rate
        warmup_steps (int): Number of warmup steps for learning rate scheduler
        label_smoothing (float): Label smoothing parameter
        pad_token_id (int): Padding token ID
        bos_token_id (int): Beginning of sequence token ID
        eos_token_id (int): End of sequence token ID
    """
    
    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.learning_rate > 0, "learning_rate must be positive"
    
    @property
    def d_k(self) -> int:
        """Dimension of query/key vectors (d_model / n_heads)."""
        return self.d_model // self.n_heads
    
    @property
    def d_v(self) -> int:
        """Dimension of value vectors (d_model / n_heads)."""
        return self.d_model // self.n_heads
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def transformer_base(cls):
        """Return configuration for Transformer-Base model."""
        return cls(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            max_seq_length=512,
            dropout=0.1
        )
    
    @classmethod
    def transformer_small(cls):
        """Return configuration for a smaller Transformer model (for testing/demo)."""
        return cls(
            vocab_size=5000,
            d_model=256,
            n_heads=4,
            n_layers=3,
            d_ff=1024,
            max_seq_length=128,
            dropout=0.1
        )
    
    @classmethod
    def transformer_large(cls):
        """Return configuration for Transformer-Large model."""
        return cls(
            vocab_size=32000,
            d_model=1024,
            n_heads=16,
            n_layers=12,
            d_ff=4096,
            max_seq_length=512,
            dropout=0.1
        )


if __name__ == "__main__":
    # Demo: Create and display different configurations
    print("Transformer-Small Configuration:")
    small_config = TransformerConfig.transformer_small()
    print(json.dumps(small_config.to_dict(), indent=2))
    print(f"d_k (key dimension): {small_config.d_k}")
    print(f"d_v (value dimension): {small_config.d_v}")
    
    print("\n" + "="*50 + "\n")
    
    print("Transformer-Base Configuration:")
    base_config = TransformerConfig.transformer_base()
    print(json.dumps(base_config.to_dict(), indent=2))
    print(f"d_k (key dimension): {base_config.d_k}")
    print(f"d_v (value dimension): {base_config.d_v}")
