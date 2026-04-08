"""
Training Script for Transformer Model

This script demonstrates how to train the Transformer model on a simple
sequence-to-sequence task (e.g., copy task, translation, or text generation).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.transformer import Transformer
from src.config import TransformerConfig
from src.utils import (
    LabelSmoothingCrossEntropy,
    WarmupScheduler,
    compute_accuracy,
    compute_perplexity,
    save_checkpoint,
    SimpleTokenizer
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
     format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CopyTaskDataset(Dataset):
    """
    Simple copy task dataset for demonstration.
    
    The task is to copy the input sequence to the output.
    This is a simple task to verify the model can learn.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 20,
        vocab_size: int = 100,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Generate random sequences
        self.data = []
        for _ in range(num_samples):
            # Random length between 5 and seq_length
            length = torch.randint(5, seq_length, (1,)).item()
            
            # Generate random sequence (excluding special tokens)
            seq = torch.randint(3, vocab_size, (length,))
            
            self.data.append(seq)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        
        # Source: original sequence
        src = seq
        
        # Target: same sequence (copy task)
        tgt_input = torch.cat([torch.tensor([self.bos_token_id]), seq[:-1]])
        tgt_output = seq
        
        return src, tgt_input, tgt_output
    
    @staticmethod
    def collate_fn(batch, pad_token_id=0):
        """Collate function to pad sequences in a batch."""
        srcs, tgt_inputs, tgt_outputs = zip(*batch)
        
        # Pad sequences
        src_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_token_id)
        tgt_input_padded = nn.utils.rnn.pad_sequence(tgt_inputs, batch_first=True, padding_value=pad_token_id)
        tgt_output_padded = nn.utils.rnn.pad_sequence(tgt_outputs, batch_first=True, padding_value=pad_token_id)
        
        return src_padded, tgt_input_padded, tgt_output_padded


def train_epoch(
    model: Transformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupScheduler,
    device: str,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (src, tgt_input, tgt_output) in enumerate(progress_bar):
        # Move to device
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output, _ = model(src, tgt_input)
        
        # Compute loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        # Compute metrics
        accuracy = compute_accuracy(output, tgt_output, ignore_index=0)
        
        # Update statistics
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}',
            'lr': f'{scheduler.get_lr():.6f}'
        })
    
    # Compute epoch statistics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    perplexity = compute_perplexity(avg_loss)
    
    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'perplexity': perplexity
    }


def evaluate(
    model: Transformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> dict:
    """Evaluate model on validation/test set."""
    model.eval()
    
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt_input, tgt_output in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            
            # Forward pass
            output, _ = model(src, tgt_input)
            
            # Compute loss
            loss = criterion(output, tgt_output)
            
            # Compute metrics
            accuracy = compute_accuracy(output, tgt_output, ignore_index=0)
            
            # Update statistics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
    
    # Compute statistics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    perplexity = compute_perplexity(avg_loss)
    
    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'perplexity': perplexity
    }


def main(args):
    """Main training function."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create configuration
    if args.model_size == 'small':
        config = TransformerConfig.transformer_small()
    elif args.model_size == 'base':
        config = TransformerConfig.transformer_base()
    else:
        config = TransformerConfig()
    
    # Override config with command line arguments
    if args.d_model:
        config.d_model = args.d_model
    if args.n_layers:
        config.n_layers = args.n_layers
    if args.n_heads:
        config.n_heads = args.n_heads
    
    logger.info(f"Model configuration: {config.to_dict()}")
    
    # Create model
    model = Transformer(config).to(device)
    param_counts = model.count_parameters()
    logger.info(f"Model parameters: {param_counts['total']:,}")
    
    # Create datasets
    train_dataset = CopyTaskDataset(
        num_samples=args.train_samples,
        seq_length=args.seq_length,
        vocab_size=config.vocab_size
    )
    
    val_dataset = CopyTaskDataset(
        num_samples=args.val_samples,
        seq_length=args.seq_length,
        vocab_size=config.vocab_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: CopyTaskDataset.collate_fn(x, pad_token_id=config.pad_token_id),
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: CopyTaskDataset.collate_fn(x, pad_token_id=config.pad_token_id),
        num_workers=args.num_workers
    )
    
    # Create loss function
    criterion = LabelSmoothingCrossEntropy(
        smoothing=config.label_smoothing,
        ignore_index=config.pad_token_id
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Create learning rate scheduler
    scheduler = WarmupScheduler(
        optimizer,
        d_model=config.d_model,
        warmup_steps=args.warmup_steps
    )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_stats = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        
        logger.info(
            f"Train - Loss: {train_stats['loss']:.4f}, "
            f"Accuracy: {train_stats['accuracy']:.4f}, "
            f"Perplexity: {train_stats['perplexity']:.2f}"
        )
        
        # Evaluate
        val_stats = evaluate(model, val_loader, criterion, device)
        
        logger.info(
            f"Val   - Loss: {val_stats['loss']:.4f}, "
            f"Accuracy: {val_stats['accuracy']:.4f}, "
            f"Perplexity: {val_stats['perplexity']:.2f}"
        )
        
        # Save checkpoint
        if epoch % args.save_every == 0 or val_stats['loss'] < best_val_loss:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"checkpoint_epoch_{epoch}.pt"
            )
            save_checkpoint(
                model, optimizer, epoch, val_stats['loss'],
                checkpoint_path, config.to_dict()
            )
            
            if val_stats['loss'] < best_val_loss:
                best_val_loss = val_stats['loss']
                best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                save_checkpoint(
                    model, optimizer, epoch, val_stats['loss'],
                    best_checkpoint_path, config.to_dict()
                )
                logger.info(f"New best model saved! (loss: {best_val_loss:.4f})")
    
    logger.info("\nTraining completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer model")
    
    # Model arguments
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'base', 'custom'],
                        help='Predefined model size')
    parser.add_argument('--d-model', type=int, default=None,
                        help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=None,
                        help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=None,
                        help='Number of attention heads')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--warmup-steps', type=int, default=4000,
                        help='Number of warmup steps for learning rate')
    
    # Data arguments
    parser.add_argument('--train-samples', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=1000,
                        help='Number of validation samples')
    parser.add_argument('--seq-length', type=int, default=20,
                        help='Maximum sequence length')
    
    # System arguments
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)
