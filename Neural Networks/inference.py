"""
Inference Script for Transformer Model

This script provides text generation capabilities using a trained Transformer model.
Supports interactive mode and batch inference.
"""

import torch
import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.transformer import Transformer
from src.config import TransformerConfig
from src.utils import load_checkpoint, SimpleTokenizer


class TransformerInference:
    """
    Wrapper class for Transformer inference.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create config
        if 'config' in checkpoint:
            self.config = TransformerConfig.from_dict(checkpoint['config'])
        else:
            print("Warning: No config found in checkpoint, using default")
            self.config = TransformerConfig.transformer_small()
        
        # Create model
        self.model = Transformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Parameters: {self.model.count_parameters()['total']:,}")
    
    def encode_input(self, text: str) -> torch.Tensor:
        """
        Encode input text to token IDs.
        
        For demonstration, we use simple character-level encoding.
        In production, use a proper tokenizer (BPE, SentencePiece, etc.).
        
        Args:
            text: Input text string
        
        Returns:
            Tensor of token IDs
        """
        # Simple character-level encoding
        tokens = [min(ord(c), self.config.vocab_size - 1) for c in text]
        return torch.tensor([tokens], dtype=torch.long, device=self.device)
    
    def decode_output(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Tensor of token IDs
        
        Returns:
            Decoded text string
        """
        # Simple character-level decoding
        ids = token_ids.cpu().squeeze().tolist()
        
        # Remove special tokens
        ids = [id for id in ids if id not in [
            self.config.pad_token_id,
            self.config.bos_token_id,
            self.config.eos_token_id
        ]]
        
        # Decode
        text = ''.join([chr(min(id, 127)) for id in ids if id < 128])
        return text
    
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> str:
        """
        Generate text given a prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text
        """
        with torch.no_grad():
            # Encode input
            src = self.encode_input(prompt)
            
            # Generate
            output = self.model.generate(
                src,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.config.eos_token_id
            )
            
            # Decode output
            generated_text = self.decode_output(output)
            
            return generated_text
    
    def interactive_mode(self):
        """Run interactive generation mode."""
        print("\n" + "="*50)
        print("Interactive Text Generation")
        print("="*50)
        print("\nCommands:")
        print("  - Type text to generate")
        print("  - '/temp X' to set temperature (e.g., /temp 0.8)")
        print("  - '/len X' to set max length (e.g., /len 100)")
        print("  - '/quit' to exit")
        print("\n" + "="*50 + "\n")
        
        temperature = 1.0
        max_length = 50
        
        while True:
            try:
                # Get input
                user_input = input("Prompt: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input == '/quit':
                    print("Goodbye!")
                    break
                elif user_input.startswith('/temp'):
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"Temperature set to {temperature}")
                    except:
                        print("Invalid temperature. Usage: /temp 0.8")
                    continue
                elif user_input.startswith('/len'):
                    try:
                        max_length = int(user_input.split()[1])
                        print(f"Max length set to {max_length}")
                    except:
                        print("Invalid length. Usage: /len 100")
                    continue
                
                # Generate text
                print("\nGenerating...")
                generated = self.generate(
                    user_input,
                    max_length=max_length,
                    temperature=temperature
                )
                
                print(f"\nGenerated: {generated}\n")
                print("-"*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main(args):
    """Main inference function."""
    # Initialize inference engine
    inference = TransformerInference(
        model_path=args.model_path,
        device='cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    
    if args.interactive:
        # Interactive mode
        inference.interactive_mode()
    else:
        # Single generation
        if not args.prompt:
            print("Error: --prompt is required for non-interactive mode")
            return
        
        print(f"Prompt: {args.prompt}")
        print("\nGenerating...")
        
        generated = inference.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        print(f"\nGenerated: {generated}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Transformer model")
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Generation arguments
    parser.add_argument('--prompt', type=str, default=None,
                        help='Input prompt for generation')
    parser.add_argument('--max-length', type=int, default=50,
                        help='Maximum length of generated sequence')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=None,
                        help='Nucleus (top-p) sampling')
    
    # Mode arguments
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    # System arguments
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    main(args)
