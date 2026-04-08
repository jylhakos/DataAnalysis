"""
Position-wise Feed-Forward Network for Transformer Architecture

Implements the feed-forward network applied to each position independently
and identically. This consists of two linear transformations with a ReLU
activation in between.

Mathematical formulation:
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Applies two linear transformations with a ReLU activation in between.
    The same network is applied to each position independently.
    
    Mathematical Formula:
        FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    In the original paper, the inner dimension d_ff = 2048 for the base model,
    which is typically 4 times the model dimension (d_model = 512).
    
    Args:
        d_model: Dimensionality of input and output
        d_ff: Dimensionality of inner layer (typically 4 * d_model)
        dropout: Dropout probability
        activation: Activation function ('relu' or 'gelu')
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super(PositionWiseFeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Choose activation function
        self.activation = activation
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'gelu':
            self.activation_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation
        # Shape: (batch_size, seq_len, d_ff)
        x = self.linear1(x)
        
        # Apply activation function
        x = self.activation_fn(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Second linear transformation
        # Shape: (batch_size, seq_len, d_model)
        x = self.linear2(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    Used in many modern transformer implementations (e.g., BERT, GPT-2).
    Provides a smooth approximation to ReLU.
    
    Mathematical Formula:
        GELU(x) = x * Φ(x)
        
    where Φ(x) is the cumulative distribution function of the standard
    normal distribution.
    
    Approximation:
        GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
    """
    
    def __init__(self, approximate: bool = True):
        super(GELU, self).__init__()
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.
        
        Args:
            x: Input tensor
        
        Returns:
            Activated tensor
        """
        if self.approximate:
            # Fast approximation
            return 0.5 * x * (
                1.0 + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        else:
            # Exact implementation using error function
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


if __name__ == "__main__":
    import math
    
    print("Testing Position-wise Feed-Forward Network\n")
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize feed-forward network
    ffn_relu = PositionWiseFeedForward(d_model, d_ff, activation='relu')
    ffn_gelu = PositionWiseFeedForward(d_model, d_ff, activation='gelu')
    
    # Forward pass
    output_relu = ffn_relu(x)
    output_gelu = ffn_gelu(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (ReLU): {output_relu.shape}")
    print(f"Output shape (GELU): {output_gelu.shape}")
    print(f"\nNumber of parameters: {sum(p.numel() for p in ffn_relu.parameters()):,}")
    
    # Compare activations
    print("\n" + "="*50)
    print("Comparing ReLU and GELU activations\n")
    
    test_input = torch.linspace(-3, 3, 100)
    relu_output = F.relu(test_input)
    gelu_output = F.gelu(test_input)
    
    print(f"Input range: [{test_input.min():.2f}, {test_input.max():.2f}]")
    print(f"ReLU output range: [{relu_output.min():.2f}, {relu_output.max():.2f}]")
    print(f"GELU output range: [{gelu_output.min():.2f}, {gelu_output.max():.2f}]")
    
    # Test specific values
    print("\nActivation values at specific points:")
    for val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        x_val = torch.tensor([val])
        relu_val = F.relu(x_val).item()
        gelu_val = F.gelu(x_val).item()
        print(f"  x={val:5.1f}: ReLU={relu_val:6.3f}, GELU={gelu_val:6.3f}")
    
    # Memory and computation comparison
    print("\n" + "="*50)
    print("Performance Characteristics:\n")
    
    print("ReLU:")
    print("  - Pros: Computationally efficient, simple gradient")
    print("  - Cons: Dead neurons (gradient=0 for x<0)")
    
    print("\nGELU:")
    print("  - Pros: Smooth, non-zero gradients everywhere, better performance in some tasks")
    print("  - Cons: Slightly more computationally expensive")
