"""
Tests for visualization functions.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualize_results import (
    plot_monte_carlo_points,
    plot_score_distribution,
    plot_mcts_tree_stats
)


class TestVisualizationFunctions:
    """Test suite for visualization functions."""
    
    def setup_method(self):
        """Setup for each test."""
        np.random.seed(42)
        self.test_results_dir = Path("test_results")
        self.test_results_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Clean up test results
        import shutil
        if self.test_results_dir.exists():
            shutil.rmtree(self.test_results_dir)
    
    @patch('visualize_results.plt.savefig')
    @patch('visualize_results.plt.close')
    def test_plot_monte_carlo_points(self, mock_close, mock_savefig):
        """Test Monte Carlo points visualization."""
        save_path = str(self.test_results_dir / "test_mc_points.png")
        
        plot_monte_carlo_points(n_points=1000, save_path=save_path)
        
        # Verify savefig was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('visualize_results.plt.savefig')
    @patch('visualize_results.plt.close')
    def test_plot_score_distribution(self, mock_close, mock_savefig):
        """Test score distribution visualization."""
        scores = [75.0, 82.0, 68.0, 91.0, 79.0, 85.0, 73.0, 88.0, 76.0, 81.0]
        save_path = str(self.test_results_dir / "test_scores.png")
        
        plot_score_distribution(scores, save_path=save_path)
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('visualize_results.plt.savefig')
    @patch('visualize_results.plt.close')
    def test_plot_score_distribution_statistics(self, mock_close, mock_savefig):
        """Test that score distribution handles statistics correctly."""
        scores = list(range(0, 101, 10))  # [0, 10, 20, ..., 100]
        save_path = str(self.test_results_dir / "test_scores_stats.png")
        
        # Should not raise any errors
        plot_score_distribution(scores, title="Test Distribution", save_path=save_path)
        
        mock_savefig.assert_called_once()
    
    @patch('visualize_results.plt.savefig')
    @patch('visualize_results.plt.close')
    def test_plot_mcts_tree_stats_file_not_found(self, mock_close, mock_savefig):
        """Test MCTS visualization when results file doesn't exist."""
        results_path = str(self.test_results_dir / "nonexistent.json")
        save_path = str(self.test_results_dir / "test_mcts.png")
        
        # Should handle missing file gracefully
        plot_mcts_tree_stats(results_path=results_path, save_path=save_path)
        
        # Should not save figure if file not found
        mock_savefig.assert_not_called()
    
    @patch('visualize_results.plt.savefig')
    @patch('visualize_results.plt.close')
    def test_plot_mcts_tree_stats_with_data(self, mock_close, mock_savefig):
        """Test MCTS visualization with actual data."""
        # Create mock results file
        results_path = self.test_results_dir / "mcts_results.json"
        save_path = str(self.test_results_dir / "test_mcts.png")
        
        mock_results = {
            'problem': 'Test problem',
            'initial_answer': 'Initial',
            'best_answer': 'Best answer',
            'best_reward': 85.5,
            'visits': 10,
            'path': ['Step 1', 'Step 2', 'Step 3']
        }
        
        with open(results_path, 'w') as f:
            json.dump(mock_results, f)
        
        plot_mcts_tree_stats(results_path=str(results_path), save_path=save_path)
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_plot_creates_directory(self):
        """Test that plot functions create output directory if needed."""
        nested_dir = self.test_results_dir / "nested" / "path"
        save_path = str(nested_dir / "test.png")
        
        with patch('visualize_results.plt.savefig'), \
             patch('visualize_results.plt.close'):
            plot_monte_carlo_points(n_points=100, save_path=save_path)
        
        # Directory should be created
        assert nested_dir.parent.exists()


class TestVisualizationDataValidation:
    """Test data validation in visualization functions."""
    
    @patch('visualize_results.plt.savefig')
    @patch('visualize_results.plt.close')
    def test_empty_score_list(self, mock_close, mock_savefig):
        """Test handling of empty score list."""
        scores = []
        
        # Should handle empty list without crashing
        try:
            plot_score_distribution(scores, save_path="test.png")
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert isinstance(e, (ValueError, ZeroDivisionError))
    
    @patch('visualize_results.plt.savefig')
    @patch('visualize_results.plt.close')
    def test_single_score(self, mock_close, mock_savefig):
        """Test handling of single score."""
        scores = [75.0]
        
        # Should handle single value
        plot_score_distribution(scores, save_path="test.png")
        
        # Should still create plot
        assert mock_savefig.called or mock_close.called
    
    @patch('visualize_results.plt.savefig')
    @patch('visualize_results.plt.close')
    def test_extreme_score_values(self, mock_close, mock_savefig):
        """Test handling of extreme score values."""
        scores = [0.0, 100.0, 50.0]
        
        # Should handle extreme values
        plot_score_distribution(scores, save_path="test.png")
        
        mock_savefig.assert_called_once()


class TestVisualizationIntegration:
    """Integration tests for visualization module."""
    
    def setup_method(self):
        """Setup for each test."""
        np.random.seed(42)
    
    @patch('visualize_results.plt.savefig')
    @patch('visualize_results.plt.close')
    @patch('visualize_results.plot_mcts_tree_stats')
    def test_generate_all_visualizations(self, mock_mcts, mock_close, mock_savefig):
        """Test that generate_all_visualizations runs without errors."""
        from visualize_results import generate_all_visualizations
        
        # Should run without raising exceptions
        generate_all_visualizations()
        
        # Should call savefig multiple times (for different plots)
        assert mock_savefig.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
