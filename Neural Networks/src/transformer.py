"""
Complete Transformer Architecture Implementation

This module implements the full Transformer model as described in
"Attention Is All You Need" (Vaswani et al., 2017).

The Transformer consists of:
- Input embeddings with positional encoding
- Encoder stack (N layers)
- Decoder stack (N layers)
- Output linear projection and softmax
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import TransformerConfig
from .embeddings import TransformerEmbedding
from .encoder import Encoder
from .decoder import Decoder
from .attention import create_padding_mask, create_decoder_mask


class Transformer(nn.Module):
    """
    Complete Transformer Model for Sequence-to-Sequence tasks.
    
    Architecture:
    1. Source and target embeddings with positional encoding
    2. Encoder: Processes source sequence
    3. Decoder: Generates target sequence while attending to encoder output
    4. Output projection: Maps decoder output to vocabulary logits
    
    Args:
        config: TransformerConfig object with model hyperparameters
    """
    
    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()
        
        self.config = config
        
        # Source and target embeddings
        self.src_embedding = TransformerEmbedding(
            config.vocab_size,
            config.d_model,
            config.max_seq_length,
            config.dropout
        )
        
        self.tgt_embedding = TransformerEmbedding(
            config.vocab_size,
            config.d_model,
            config.max_seq_length,
            config.dropout
        )
        
        # Encoder
        self.encoder = Encoder(
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.d_ff,
            config.dropout
        )
        
        # Decoder
        self.decoder = Decoder(
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.d_ff,
            config.dropout
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize model parameters using Xavier uniform initialization.
        
        This helps with gradient flow and convergence.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Encode source sequence.
        
        Args:
            src: Source token IDs of shape (batch_size, src_seq_len)
            src_mask: Optional source padding mask
        
        Returns:
            encoder_output: Encoded representations (batch_size, src_seq_len, d_model)
            attention_weights: List of attention weights from each encoder layer
        """
        # Embed source tokens
        src_embedded = self.src_embedding(src)
        
        # Encode
        encoder_output, attention_weights = self.encoder(src_embedded, src_mask)
        
        return encoder_output, attention_weights
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list, list]:
        """
        Decode target sequence given encoder output.
        
        Args:
            tgt: Target token IDs of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)
            tgt_mask: Optional target mask (causal + padding)
            src_mask: Optional source padding mask
        
        Returns:
            decoder_output: Decoded representations (batch_size, tgt_seq_len, d_model)
            self_attn_weights: List of self-attention weights
            cross_attn_weights: List of cross-attention weights
        """
        # Embed target tokens
        tgt_embedded = self.tgt_embedding(tgt)
        
        # Decode
        decoder_output, self_attn_weights, cross_attn_weights = self.decoder(
            tgt_embedded, encoder_output, tgt_mask, src_mask
        )
        
        return decoder_output, self_attn_weights, cross_attn_weights
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through the complete Transformer.
        
        Args:
            src: Source token IDs (batch_size, src_seq_len)
            tgt: Target token IDs (batch_size, tgt_seq_len)
            src_mask: Optional source padding mask
            tgt_mask: Optional target mask
        
        Returns:
            output: Logits over vocabulary (batch_size, tgt_seq_len, vocab_size)
            attention_weights: Dictionary containing all attention weights
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, self.config.pad_token_id)
        
        if tgt_mask is None:
            tgt_mask = create_decoder_mask(tgt, self.config.pad_token_id)
        
        # Encode source sequence
        encoder_output, enc_attn_weights = self.encode(src, src_mask)
        
        # Decode target sequence
        decoder_output, dec_self_attn_weights, dec_cross_attn_weights = self.decode(
            tgt, encoder_output, tgt_mask, src_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        # Collect all attention weights
        attention_weights = {
            'encoder_self_attention': enc_attn_weights,
            'decoder_self_attention': dec_self_attn_weights,
            'decoder_cross_attention': dec_cross_attn_weights
        }
        
        return output, attention_weights
    
    def generate(
        self,
        src: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate target sequence autoregressively given source sequence.
        
        Uses greedy decoding, top-k sampling, or nucleus (top-p) sampling.
        
        Args:
            src: Source token IDs (batch_size, src_seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
            top_p: If set, use nucleus sampling with cumulative probability p
            eos_token_id: End-of-sequence token ID
        
        Returns:
            Generated token IDs (batch_size, generated_seq_len)
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        # Encode source
        src_mask = create_padding_mask(src, self.config.pad_token_id)
        encoder_output, _ = self.encode(src, src_mask)
        
        # Initialize target with BOS token
        tgt = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one at a time
        for _ in range(max_length - 1):
            # Create target mask
            tgt_mask = create_decoder_mask(tgt, self.config.pad_token_id)
            
            # Decode
            decoder_output, _, _ = self.decode(tgt, encoder_output, tgt_mask, src_mask)
            
            # Project to vocabulary (only need last position)
            logits = self.output_projection(decoder_output[:, -1, :])  # (batch_size, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # Update finished sequences
            finished |= (next_token.squeeze(-1) == eos_token_id)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return tgt
    
    def count_parameters(self) -> dict:
        """
        Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        embedding_params = sum(p.numel() for p in self.src_embedding.parameters()) + \
                          sum(p.numel() for p in self.tgt_embedding.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'encoder': encoder_params,
            'decoder': decoder_params,
            'embeddings': embedding_params,
            'output_projection': sum(p.numel() for p in self.output_projection.parameters())
        }


if __name__ == "__main__":
    print("Testing Complete Transformer Model\n")
    
    # Create configuration
    config = TransformerConfig.transformer_small()
    
    print("Configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  max_seq_length: {config.max_seq_length}")
    
    # Create model
    model = Transformer(config)
    
    # Create sample data
    batch_size = 2
    src_seq_len = 15
    tgt_seq_len = 10
    
    src = torch.randint(3, config.vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(3, config.vocab_size, (batch_size, tgt_seq_len))
    
    print(f"\nInput shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")
    
    # Forward pass
    output, attention_weights = model(src, tgt)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: (batch_size={batch_size}, tgt_seq_len={tgt_seq_len}, vocab_size={config.vocab_size})")
    
    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Test generation
    print("\n" + "="*50)
    print("Testing Text Generation\n")
    
    generated = model.generate(src[:1], max_length=20, temperature=1.0)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    # Test with different sampling strategies
    print("\nTesting different sampling strategies:")
    
    # Greedy (low temperature)
    greedy = model.generate(src[:1], max_length=15, temperature=0.1)
    print(f"Greedy (temp=0.1): {greedy[0].tolist()[:10]}...")
    
    # Top-k sampling
    topk = model.generate(src[:1], max_length=15, temperature=1.0, top_k=50)
    print(f"Top-k (k=50): {topk[0].tolist()[:10]}...")
    
    # Nucleus sampling
    nucleus = model.generate(src[:1], max_length=15, temperature=1.0, top_p=0.9)
    print(f"Nucleus (p=0.9): {nucleus[0].tolist()[:10]}...")
