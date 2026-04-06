"""
LLM Training with Gradient Descent

This module demonstrates training a transformer-based language model using
gradient descent with various optimizations:
- Mini-batch gradient descent
- Gradient clipping
- Adam optimizer
- Learning rate scheduling
- Mixed precision training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils as nn_utils
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import time


class TextDataset(Dataset):
    """Simple dataset for language modeling."""
    
    def __init__(self, data: torch.Tensor, block_size: int):
        """
        Initialize dataset.
        
        Args:
            data: Token indices
            block_size: Context length
        """
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


class LLMTrainer:
    """
    Trainer for language models with gradient descent.
    
    Implements:
    - Mini-batch gradient descent
    - Gradient clipping
    - Gradient accumulation
    - Mixed precision training (optional)
    - Learning rate scheduling
    """
    
    def __init__(self, model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor,
                 config: Dict):
        """
        Initialize trainer.
        
        Args:
            model: Language model to train
            train_data: Training token indices
            val_data: Validation token indices
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Create datasets and dataloaders
        block_size = config.get('block_size', 128)
        batch_size = config.get('batch_size', 32)
        
        self.train_dataset = TextDataset(train_data, block_size)
        self.val_dataset = TextDataset(val_data, block_size)
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Optimizer
        learning_rate = config.get('learning_rate', 3e-4)
        weight_decay = config.get('weight_decay', 0.1)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = None
        if config.get('use_scheduler', True):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 10),
                eta_min=learning_rate * 0.1
            )
        
        # Gradient clipping
        self.grad_clip = config.get('grad_clip', 1.0)
        
        # Gradient accumulation
        self.accum_steps = config.get('gradient_accumulation_steps', 1)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', False)
        if self.use_amp and self.device == 'cuda':
            from torch.cuda.amp import autocast, GradScaler
            self.scaler = GradScaler()
        else:
            self.use_amp = False
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'grad_norms': [],
            'perplexity': []
        }
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, loss = self.model(x, y)
                    loss = loss / self.accum_steps
            else:
                logits, loss = self.model(x, y)
                loss = loss / self.accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update parameters (with gradient accumulation)
            if (batch_idx + 1) % self.accum_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = nn_utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip
                )
                self.history['grad_norms'].append(grad_norm.item())
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accum_steps
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Average validation loss and perplexity
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            logits, loss = self.model(x, y)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def train(self, epochs: int):
        """
        Train model for specified number of epochs.
        
        Args:
            epochs: Number of training epochs
        """
        print("=" * 70)
        print("Training Language Model with Gradient Descent")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {self.config.get('batch_size', 32)}")
        print(f"  Learning rate: {self.config.get('learning_rate', 3e-4)}")
        print(f"  Gradient clipping: {self.grad_clip}")
        print(f"  Gradient accumulation: {self.accum_steps} steps")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Device: {self.device}")
        print()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            val_loss, perplexity = self.evaluate()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.config.get('learning_rate', 3e-4)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['perplexity'].append(perplexity)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Perplexity: {perplexity:.2f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  → New best validation loss: {best_val_loss:.4f}")
        
        print("\n" + "=" * 70)
        print(f"Training Complete! Best Val Loss: {best_val_loss:.4f}")
        print("=" * 70)
    
    def plot_training_history(self, save_path: str = 'visualizations/llm_training.png'):
        """
        Plot training metrics.
        
        Args:
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=13)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot perplexity
        axes[0, 1].plot(epochs, self.history['perplexity'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Perplexity', fontsize=11)
        axes[0, 1].set_title('Validation Perplexity', fontsize=13)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot learning rate
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=13)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Plot gradient norms
        if self.history['grad_norms']:
            axes[1, 1].plot(self.history['grad_norms'], 'c-', alpha=0.5, linewidth=0.5)
            axes[1, 1].axhline(y=self.grad_clip, color='r', linestyle='--', 
                             linewidth=2, label=f'Clip Threshold ({self.grad_clip})')
            axes[1, 1].set_xlabel('Optimization Step', fontsize=11)
            axes[1, 1].set_ylabel('Gradient Norm', fontsize=11)
            axes[1, 1].set_title('Gradient Norms (with Clipping)', fontsize=13)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nTraining history plot saved to: {save_path}")


def create_synthetic_data(vocab_size: int, n_samples: int) -> torch.Tensor:
    """
    Create synthetic text data for demonstration.
    
    Args:
        vocab_size: Size of vocabulary
        n_samples: Number of tokens to generate
        
    Returns:
        Token indices
    """
    # Generate random sequences with some structure
    # (In practice, use real text data)
    data = torch.randint(0, vocab_size, (n_samples,))
    return data


def demo_llm_training():
    """Demonstrate LLM training with gradient descent."""
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # Import model
    from transformer import GPTModel
    
    print("=" * 70)
    print("Language Model Training Demo")
    print("=" * 70)
    
    # Model configuration (small model for demo)
    model_config = {
        'vocab_size': 1000,
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 4,
        'd_ff': 1024,
        'max_seq_len': 256,
        'dropout': 0.1,
        'use_rmsnorm': True,
        'activation': 'gelu'
    }
    
    # Create model
    model = GPTModel(**model_config)
    n_params = model.count_parameters()
    print(f"\nModel: {n_params:,} parameters ({n_params/1e6:.2f}M)")
    
    # Create synthetic data
    print("\nGenerating synthetic training data...")
    train_data = create_synthetic_data(model_config['vocab_size'], 50000)
    val_data = create_synthetic_data(model_config['vocab_size'], 10000)
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    
    # Training configuration
    train_config = {
        'batch_size': 64,
        'block_size': 128,
        'learning_rate': 5e-4,
        'weight_decay': 0.1,
        'grad_clip': 1.0,
        'gradient_accumulation_steps': 1,
        'use_scheduler': True,
        'use_amp': False,  # Set to True if using CUDA
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 20
    }
    
    # Create trainer
    trainer = LLMTrainer(model, train_data, val_data, train_config)
    
    # Train model
    trainer.train(epochs=train_config['epochs'])
    
    # Plot training history
    trainer.plot_training_history()
    
    # Demonstrate gradient clipping effect
    print("\n" + "=" * 70)
    print("Gradient Clipping Analysis")
    print("=" * 70)
    
    grad_norms = trainer.history['grad_norms']
    if grad_norms:
        grad_norms = np.array(grad_norms)
        clipped_fraction = np.mean(grad_norms >= train_config['grad_clip'])
        
        print(f"Total optimization steps: {len(grad_norms)}")
        print(f"Steps with clipped gradients: {int(clipped_fraction * len(grad_norms))}")
        print(f"Clipping frequency: {clipped_fraction*100:.1f}%")
        print(f"Average gradient norm: {np.mean(grad_norms):.4f}")
        print(f"Max gradient norm: {np.max(grad_norms):.4f}")
        print(f"Clipping threshold: {train_config['grad_clip']}")
        
        print("\nGradient clipping prevents exploding gradients and stabilizes training!")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


def compare_optimizers():
    """Compare different gradient descent variants."""
    from transformer import GPTModel
    
    print("=" * 70)
    print("Comparing Gradient Descent Optimizers")
    print("=" * 70)
    
    # Small model for quick comparison
    model_config = {
        'vocab_size': 500,
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 4,
        'd_ff': 512,
        'max_seq_len': 128,
        'dropout': 0.1,
    }
    
    # Generate data
    train_data = create_synthetic_data(model_config['vocab_size'], 10000)
    val_data = create_synthetic_data(model_config['vocab_size'], 2000)
    
    optimizers_to_test = ['sgd', 'adam', 'adamw']
    results = {}
    
    for opt_name in optimizers_to_test:
        print(f"\nTraining with {opt_name.upper()}...")
        
        model = GPTModel(**model_config)
        
        config = {
            'batch_size': 32,
            'block_size': 64,
            'learning_rate': 1e-3 if opt_name == 'sgd' else 3e-4,
            'grad_clip': 1.0,
            'epochs': 10,
            'device': 'cpu'
        }
        
        trainer = LLMTrainer(model, train_data, val_data, config)
        
        # Replace optimizer
        if opt_name == 'sgd':
            trainer.optimizer = torch.optim.SGD(
                model.parameters(), lr=config['learning_rate']
            )
        elif opt_name == 'adam':
            trainer.optimizer = torch.optim.Adam(
                model.parameters(), lr=config['learning_rate']
            )
        # 'adamw' is already default
        
        trainer.train(epochs=config['epochs'])
        results[opt_name] = trainer.history['val_loss']
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    for opt_name, losses in results.items():
        plt.plot(losses, linewidth=2, label=opt_name.upper(), marker='o')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Optimizer Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/optimizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print("=" * 70)


if __name__ == "__main__":
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # Run main training demo
    demo_llm_training()
    
    # Optionally compare optimizers
    # compare_optimizers()
