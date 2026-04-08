"""
Ollama LLM Client

Provides a simple interface to interact with local LLMs running on Ollama.
"""

import requests
import json
from typing import List, Dict, Optional


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """
        Initialize Ollama client.
        
        Args:
            base_url (str): Base URL of Ollama server
            model (str): Model name to use (e.g., 'llama2', 'mistral', 'phi')
        """
        self.base_url = base_url
        self.model = model
        
    def check_connection(self) -> bool:
        """
        Check if Ollama server is accessible.
        
        Returns:
            bool: True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """
        List available models on the Ollama server.
        
        Returns:
            List[str]: List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                 max_tokens: int = 500) -> Optional[str]:
        """
        Generate text from the LLM.
        
        Args:
            prompt (str): Input prompt
            temperature (float): Sampling temperature (0.0 to 2.0)
                Higher values = more random, lower = more deterministic
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            Optional[str]: Generated text or None if error
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                print(f"Error: Status code {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error generating text: {e}")
            return None
    
    def generate_multiple(self, prompt: str, n_samples: int = 5,
                         temperature: float = 1.0, max_tokens: int = 500) -> List[str]:
        """
        Generate multiple responses for Monte Carlo sampling.
        
        Args:
            prompt (str): Input prompt
            n_samples (int): Number of samples to generate
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens per sample
            
        Returns:
            List[str]: List of generated responses
        """
        responses = []
        print(f"Generating {n_samples} samples at temperature {temperature}...")
        
        for i in range(n_samples):
            response = self.generate(prompt, temperature, max_tokens)
            if response:
                responses.append(response)
                print(f"  Sample {i+1}/{n_samples} generated ({len(response)} chars)")
            else:
                print(f"  Sample {i+1}/{n_samples} failed")
        
        return responses
    
    def score_response(self, response: str, criteria: str = "accuracy and clarity") -> Optional[float]:
        """
        Use the LLM to score a response on a 0-100 scale.
        
        Args:
            response (str): Response to evaluate
            criteria (str): Evaluation criteria
            
        Returns:
            Optional[float]: Score between 0-100, or None if error
        """
        scoring_prompt = f"""
Evaluate the following response based on {criteria}.
Provide ONLY a numeric score between 0 and 100, where:
- 0 = completely incorrect or nonsensical
- 50 = partially correct but with significant issues
- 100 = perfect, accurate, and clear

Response to evaluate:
{response}

Score (0-100):"""
        
        result = self.generate(scoring_prompt, temperature=0.2, max_tokens=10)
        
        if result:
            try:
                # Extract numeric value from response
                import re
                match = re.search(r'\d+', result)
                if match:
                    score = float(match.group())
                    return min(max(score, 0), 100)  # Clamp to 0-100
            except ValueError:
                pass
        
        return None


def test_ollama_connection():
    """Test function to verify Ollama connection."""
    print("Testing Ollama connection...")
    print()
    
    client = OllamaClient()
    
    # Check connection
    if not client.check_connection():
        print("ERROR: Cannot connect to Ollama server at http://localhost:11434")
        print("Please ensure Ollama is running:")
        print("  docker start ollama")
        return False
    
    print("SUCCESS: Connected to Ollama server")
    print()
    
    # List models
    models = client.list_models()
    print(f"Available models: {', '.join(models) if models else 'None'}")
    print()
    
    if not models:
        print("WARNING: No models found. Please pull a model:")
        print("  docker exec -it ollama ollama pull llama2")
        return False
    
    # Test generation
    print("Testing text generation...")
    response = client.generate("Say 'Hello, Monte Carlo!' and nothing else.", temperature=0.1)
    
    if response:
        print(f"Response: {response}")
        print()
        print("SUCCESS: LLM is working correctly")
        return True
    else:
        print("ERROR: Failed to generate response")
        return False


if __name__ == "__main__":
    test_ollama_connection()
