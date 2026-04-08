"""
Transformer Architecture Implementation for Large Language Models

This package implements a complete Transformer model from scratch,
following the architecture described in "Attention Is All You Need" (2017).
"""

__version__ = "1.0.0"
__author__ = "Neural Networks Project"

from .transformer import Transformer
from .config import TransformerConfig

__all__ = ['Transformer', 'TransformerConfig']
