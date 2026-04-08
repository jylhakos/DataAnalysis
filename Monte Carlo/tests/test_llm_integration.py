"""
Tests for LLM integration with Ollama.

Note: These tests require a running Ollama server.
Some tests will be skipped if Ollama is not available.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import OllamaClient


class TestOllamaClient:
    """Test suite for OllamaClient."""
    
    def test_client_initialization(self):
        """Test basic client initialization."""
        client = OllamaClient(base_url="http://localhost:11434", model="llama2")
        
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama2"
    
    def test_client_default_initialization(self):
        """Test client with default parameters."""
        client = OllamaClient()
        
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama2"
    
    @patch('llm_client.requests.get')
    def test_check_connection_success(self, mock_get):
        """Test connection check when server is available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        client = OllamaClient()
        result = client.check_connection()
        
        assert result is True
        mock_get.assert_called_once()
    
    @patch('llm_client.requests.get')
    def test_check_connection_failure(self, mock_get):
        """Test connection check when server is unavailable."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        client = OllamaClient()
        result = client.check_connection()
        
        assert result is False
    
    @patch('llm_client.requests.get')
    def test_list_models_success(self, mock_get):
        """Test listing models when server responds."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'name': 'llama2'},
                {'name': 'mistral'},
                {'name': 'phi'}
            ]
        }
        mock_get.return_value = mock_response
        
        client = OllamaClient()
        models = client.list_models()
        
        assert models == ['llama2', 'mistral', 'phi']
    
    @patch('llm_client.requests.get')
    def test_list_models_failure(self, mock_get):
        """Test listing models when server fails."""
        mock_get.side_effect = requests.exceptions.RequestException("Error")
        
        client = OllamaClient()
        models = client.list_models()
        
        assert models == []
    
    @patch('llm_client.requests.post')
    def test_generate_success(self, mock_post):
        """Test successful text generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Generated text response'
        }
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        result = client.generate("Test prompt", temperature=0.7, max_tokens=100)
        
        assert result == 'Generated text response'
        
        # Verify the request payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['model'] == 'llama2'
        assert payload['prompt'] == 'Test prompt'
        assert payload['options']['temperature'] == 0.7
        assert payload['options']['num_predict'] == 100
    
    @patch('llm_client.requests.post')
    def test_generate_failure(self, mock_post):
        """Test text generation failure."""
        mock_post.side_effect = requests.exceptions.RequestException("Error")
        
        client = OllamaClient()
        result = client.generate("Test prompt")
        
        assert result is None
    
    @patch('llm_client.requests.post')
    def test_generate_multiple(self, mock_post):
        """Test multiple generation samples."""
        # Mock returns different responses
        responses = [
            {'response': 'Response 1'},
            {'response': 'Response 2'},
            {'response': 'Response 3'}
        ]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = responses
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        results = client.generate_multiple("Test prompt", n_samples=3, temperature=1.0)
        
        assert len(results) == 3
        assert results == ['Response 1', 'Response 2', 'Response 3']
    
    @patch('llm_client.requests.post')
    def test_score_response_success(self, mock_post):
        """Test response scoring."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': '85'
        }
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        score = client.score_response("Good response", criteria="accuracy")
        
        assert score == 85.0
    
    @patch('llm_client.requests.post')
    def test_score_response_extraction(self, mock_post):
        """Test score extraction from various response formats."""
        test_cases = [
            ('85', 85.0),
            ('Score: 92', 92.0),
            ('The score is 78 out of 100', 78.0),
            ('100', 100.0),
            ('0', 0.0),
        ]
        
        client = OllamaClient()
        
        for response_text, expected_score in test_cases:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'response': response_text}
            mock_post.return_value = mock_response
            
            score = client.score_response("Test", criteria="test")
            assert score == expected_score, f"Failed for response: {response_text}"
    
    @patch('llm_client.requests.post')
    def test_score_response_clamping(self, mock_post):
        """Test that scores are clamped to 0-100 range."""
        client = OllamaClient()
        
        # Test upper bound
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': '150'}
        mock_post.return_value = mock_response
        
        score = client.score_response("Test")
        assert score == 100.0
        
        # Test lower bound (negative)
        mock_response.json.return_value = {'response': '-10'}
        score = client.score_response("Test")
        assert score == 0.0


class TestOllamaClientLive:
    """
    Live integration tests with actual Ollama server.
    These tests require Ollama to be running.
    """
    
    @pytest.fixture
    def ollama_available(self):
        """Check if Ollama server is available."""
        client = OllamaClient()
        return client.check_connection()
    
    def test_live_connection(self, ollama_available):
        """Test actual connection to Ollama server."""
        if not ollama_available:
            pytest.skip("Ollama server not available")
        
        client = OllamaClient()
        assert client.check_connection() is True
    
    def test_live_list_models(self, ollama_available):
        """Test listing models from live server."""
        if not ollama_available:
            pytest.skip("Ollama server not available")
        
        client = OllamaClient()
        models = client.list_models()
        
        assert isinstance(models, list)
        # If Ollama is running, it should have at least one model
        # (but this might fail if no models are pulled)
    
    def test_live_generation(self, ollama_available):
        """Test actual text generation."""
        if not ollama_available:
            pytest.skip("Ollama server not available")
        
        client = OllamaClient()
        
        # Check if models are available
        models = client.list_models()
        if not models:
            pytest.skip("No models available in Ollama")
        
        # Simple generation test
        response = client.generate(
            "Say 'test' and nothing else.",
            temperature=0.1,
            max_tokens=20
        )
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
