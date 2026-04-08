"""
Utility Functions for Transformer Training and Evaluation

Includes:
- Learning rate schedulers
- Label smoothing
- Tokenization utilities
- Checkpoint management
- Metrics computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
from typing import Optional, Dict, Any
from pathlib import Path


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Label smoothing prevents the model from becoming overconfident by
    distributing some probability mass to other classes.
    
    Mathematical Formula:
        y'_i = (1 - ε) * y_i + ε / K
    
    where ε is the smoothing parameter and K is the number of classes.
    
    Args:
        smoothing: Label smoothing parameter (typically 0.1)
        ignore_index: Index to ignore (padding token)
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.
        
        Args:
            pred: Predictions of shape (batch_size, seq_len, vocab_size)
            target: Target indices of shape (batch_size, seq_len)
        
        Returns:
            Loss value
        """
        vocab_size = pred.size(-1)
        pred = pred.view(-1, vocab_size)
        target = target.view(-1)
        
        # Create smoothed labels
        confidence = 1.0 - self.smoothing
        smoothed_target = torch.zeros_like(pred)
        smoothed_target.fill_(self.smoothing / (vocab_size - 1))
        smoothed_target.scatter_(1, target.unsqueeze(1), confidence)
        
        # Ignore padding
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).unsqueeze(1)
            smoothed_target = smoothed_target * mask
        
        # Compute log probabilities
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Compute loss
        loss = -(smoothed_target * log_probs).sum(dim=-1)
        
        # Mask out ignored indices
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).float()
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss


class WarmupScheduler:
    """
    Learning rate scheduler with linear warmup and inverse square root decay.
    
    As described in "Attention Is All You Need":
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension
        warmup_steps: Number of warmup steps
        factor: Scaling factor (default: 1.0)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0
    
    def step(self):
        """Update learning rate."""
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """
        Compute current learning rate.
        
        Returns:
            Current learning rate
        """
        step = max(1, self.step_num)  # Avoid division by zero
        lr = self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
        return lr


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0) -> float:
    """
    Compute accuracy excluding padding tokens.
    
    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        targets: Ground truth (batch_size, seq_len)
        ignore_index: Index to ignore (padding)
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Get predicted tokens
    predictions = logits.argmax(dim=-1)
    
    # Create mask for non-padding tokens
    mask = (targets != ignore_index)
    
    # Compute accuracy
    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()
    
    if total == 0:
        return 0.0
    
    return correct / total


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity = exp(loss)
    
    Args:
        loss: Cross-entropy loss value
    
    Returns:
        Perplexity value
    """
    return math.exp(min(loss, 100))  # Cap to prevent overflow


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    config: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        config: Optional model configuration dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint on
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown')}")
    
    return checkpoint


class SimpleTokenizer:
    """
    Simple character-level or word-level tokenizer for demonstration.
    
    For production use, consider using SentencePiece or BPE tokenizers.
    
    Args:
        vocab_size: Maximum vocabulary size
        pad_token: Padding token
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
        unk_token: Unknown token
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        pad_token: str = '<PAD>',
        bos_token: str = '<BOS>',
        eos_token: str = '<EOS>',
        unk_token: str = '<UNK>'
    ):
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        
        # Initialize special tokens
        self.token2id = {
            pad_token: 0,
            bos_token: 1,
            eos_token: 2,
            unk_token: 3
        }
        self.id2token = {v: k for k, v in self.token2id.items()}
        
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
        self.next_id = 4
    
    def fit(self, texts: list, level: str = 'word'):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            level: 'word' or 'char' tokenization
        """
        from collections import Counter
        
        # Count tokens
        counter = Counter()
        for text in texts:
            if level == 'word':
                tokens = text.lower().split()
            elif level == 'char':
                tokens = list(text)
            else:
                raise ValueError(f"Unsupported level: {level}")
            
            counter.update(tokens)
        
        # Add most common tokens to vocabulary
        for token, _ in counter.most_common(self.vocab_size - 4):
            if token not in self.token2id:
                self.token2id[token] = self.next_id
                self.id2token[self.next_id] = token
                self.next_id += 1
    
    def encode(self, text: str, level: str = 'word', add_special_tokens: bool = True) -> list:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
            level: 'word' or 'char' tokenization
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        if level == 'word':
            tokens = text.lower().split()
        elif level == 'char':
            tokens = list(text)
        else:
            raise ValueError(f"Unsupported level: {level}")
        
        # Convert tokens to IDs
        ids = [self.token2id.get(token, self.unk_token_id) for token in tokens]
        
        # Add special tokens
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        return ids
    
    def decode(self, ids: list, skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text string
        """
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        tokens = []
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            tokens.append(self.id2token.get(id, self.unk_token))
        
        return ' '.join(tokens)
    
    def save(self, filepath: str):
        """Save tokenizer vocabulary."""
        data = {
            'token2id': self.token2id,
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'pad': self.pad_token,
                'bos': self.bos_token,
                'eos': self.eos_token,
                'unk': self.unk_token
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer vocabulary."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.token2id = data['token2id']
        tokenizer.id2token = {int(k): v for k, v in tokenizer.token2id.items()}
        
        return tokenizer


if __name__ == "__main__":
    print("Testing Utility Functions\n")
    
    # Test label smoothing
    print("="*50)
    print("Testing Label Smoothing\n")
    
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, ignore_index=0)
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = criterion(logits, targets)
    print(f"Loss with label smoothing: {loss.item():.4f}")
    
    # Test learning rate scheduler
    print("\n" + "="*50)
    print("Testing Learning Rate Scheduler\n")
    
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    
    scheduler = WarmupScheduler(optimizer, d_model=512, warmup_steps=4000)
    
    print("Learning rates for first few steps:")
    for step in range(10):
        scheduler.step()
        print(f"  Step {step+1}: lr = {scheduler.get_lr():.6f}")
    
    # Test accuracy computation
    print("\n" + "="*50)
    print("Testing Accuracy Computation\n")
    
    logits = torch.randn(2, 10, 100)
    targets = torch.randint(0, 100, (2, 10))
    targets[0, -2:] = 0  # Add some padding
    
    accuracy = compute_accuracy(logits, targets, ignore_index=0)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Test perplexity
    print("\n" + "="*50)
    print("Testing Perplexity Computation\n")
    
    for loss_val in [0.5, 1.0, 2.0, 3.0, 5.0]:
        perplexity = compute_perplexity(loss_val)
        print(f"  Loss: {loss_val:.2f} → Perplexity: {perplexity:.2f}")
    
    # Test simple tokenizer
    print("\n" + "="*50)
    print("Testing Simple Tokenizer\n")
    
    texts = [
        "hello world this is a test",
        "machine learning is amazing",
        "transformers are powerful"
    ]
    
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.fit(texts, level='word')
    
    print(f"Vocabulary size: {len(tokenizer.token2id)}")
    print(f"Sample vocabulary: {list(tokenizer.token2id.keys())[:10]}")
    
    text = "hello world"
    encoded = tokenizer.encode(text, level='word')
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal text: '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
