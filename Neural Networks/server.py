"""
Simple Inference Server for Transformer Model

This script creates a REST API server for text generation using Flask.
Provides endpoints for health checks, text generation, and encoding.
"""

import torch
import argparse
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.transformer import Transformer
from src.config import TransformerConfig


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model instance
model = None
config = None
device = None


def load_model(model_path: str, device_name: str = 'cpu'):
    """Load model into global variable."""
    global model, config, device
    
    device = torch.device(device_name)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create config
    if 'config' in checkpoint:
        config = TransformerConfig.from_dict(checkpoint['config'])
    else:
        logger.warning("No config found in checkpoint, using default")
        config = TransformerConfig.transformer_small()
    
    # Create and load model
    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Parameters: {model.count_parameters()['total']:,}")


def encode_text(text: str) -> torch.Tensor:
    """Simple character-level encoding."""
    tokens = [min(ord(c), config.vocab_size - 1) for c in text]
    return torch.tensor([tokens], dtype=torch.long, device=device)


def decode_tokens(token_ids: torch.Tensor) -> str:
    """Simple character-level decoding."""
    ids = token_ids.cpu().squeeze().tolist()
    
    # Remove special tokens
    ids = [id for id in ids if id not in [
        config.pad_token_id,
        config.bos_token_id,
        config.eos_token_id
    ]]
    
    # Decode
    text = ''.join([chr(min(id, 127)) for id in ids if id < 128])
    return text


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    param_counts = model.count_parameters()
    
    return jsonify({
        'config': config.to_dict(),
        'parameters': param_counts,
        'device': str(device)
    })


@app.route('/generate', methods=['POST'])
def generate():
    """
    Text generation endpoint.
    
    Expects JSON with:
        - prompt: Input text
        - max_length: Maximum generation length (optional, default: 50)
        - temperature: Sampling temperature (optional, default: 1.0)
        - top_k: Top-k sampling (optional)
        - top_p: Nucleus sampling (optional)
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Parse request
        data = request.get_json()
        
        if 'prompt' not in data:
            return jsonify({'error': 'Missing prompt field'}), 400
        
        prompt = data['prompt']
        max_length = data.get('max_length', 50)
        temperature = data.get('temperature', 1.0)
        top_k = data.get('top_k', None)
        top_p = data.get('top_p', None)
        
        # Validate parameters
        if max_length > config.max_seq_length:
            max_length = config.max_seq_length
        
        if temperature <= 0:
            return jsonify({'error': 'Temperature must be positive'}), 400
        
        # Generate text
        with torch.no_grad():
            src = encode_text(prompt)
            
            output = model.generate(
                src,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=config.eos_token_id
            )
            
            generated_text = decode_tokens(output)
        
        return jsonify({
            'prompt': prompt,
            'generated_text': generated_text,
            'parameters': {
                'max_length': max_length,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            }
        })
    
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/encode', methods=['POST'])
def encode():
    """
    Encoding endpoint - returns embeddings for input text.
    
    Expects JSON with:
        - text: Input text
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Parse request
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        
        # Encode text
        with torch.no_grad():
            src = encode_text(text)
            
            # Get encoder output
            encoder_output, _ = model.encode(src)
            
            # Convert to list for JSON serialization
            embeddings = encoder_output.cpu().tolist()
        
        return jsonify({
            'text': text,
            'embeddings': embeddings,
            'shape': list(encoder_output.shape)
        })
    
    except Exception as e:
        logger.error(f"Error during encoding: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_generate', methods=['POST'])
def batch_generate():
    """
    Batch generation endpoint.
    
    Expects JSON with:
        - prompts: List of input prompts
        - max_length: Maximum generation length (optional)
        - temperature: Sampling temperature (optional)
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Parse request
        data = request.get_json()
        
        if 'prompts' not in data:
            return jsonify({'error': 'Missing prompts field'}), 400
        
        prompts = data['prompts']
        max_length = data.get('max_length', 50)
        temperature = data.get('temperature', 1.0)
        
        if not isinstance(prompts, list):
            return jsonify({'error': 'prompts must be a list'}), 400
        
        # Generate for each prompt
        results = []
        for prompt in prompts:
            with torch.no_grad():
                src = encode_text(prompt)
                output = model.generate(
                    src,
                    max_length=max_length,
                    temperature=temperature,
                    eos_token_id=config.eos_token_id
                )
                generated_text = decode_tokens(output)
            
            results.append({
                'prompt': prompt,
                'generated_text': generated_text
            })
        
        return jsonify({
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        logger.error(f"Error during batch generation: {str(e)}")
        return jsonify({'error': str(e)}), 500


def main(args):
    """Start the server."""
    # Load model
    device_name = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    load_model(args.model_path, device_name)
    
    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference server")
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Server arguments
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--port', type=int, default=5000,
                        help='Server port')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    # System arguments
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    main(args)
