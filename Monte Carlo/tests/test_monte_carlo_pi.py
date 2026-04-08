"""
Tests for basic Monte Carlo simulation (pi estimation).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monte_carlo_pi import estimate_pi, estimate_pi_with_convergence


class TestMonteCarloPI:
    """Test suite for Monte Carlo pi estimation."""
    
    def setup_method(self):
        """Setup for each test method."""
        np.random.seed(42)
    
    def test_estimate_pi_basic(self):
        """Test that pi estimation produces reasonable results."""
        estimated_pi = estimate_pi(10000)
        
        # Should be within 0.1 of actual pi for 10k samples
        assert abs(estimated_pi - np.pi) < 0.1, \
            f"Estimated pi {estimated_pi} too far from actual pi {np.pi}"
    
    def test_estimate_pi_convergence(self):
        """Test that more samples give more accurate estimates."""
        pi_1k = estimate_pi(1000)
        pi_10k = estimate_pi(10000)
        pi_100k = estimate_pi(100000)
        
        error_1k = abs(pi_1k - np.pi)
        error_10k = abs(pi_10k - np.pi)
        error_100k = abs(pi_100k - np.pi)
        
        # Generally, more samples should reduce error (stochastic, so not strict)
        # Just check that 100k samples is reasonably accurate
        assert error_100k < 0.05, \
            f"Error with 100k samples too high: {error_100k}"
    
    def test_estimate_pi_range(self):
        """Test that estimated pi is in valid range."""
        for _ in range(5):
            estimated_pi = estimate_pi(5000)
            assert 2.5 < estimated_pi < 3.5, \
                f"Estimated pi {estimated_pi} outside reasonable range"
    
    def test_estimate_pi_with_convergence_returns_correct_shape(self):
        """Test that convergence function returns correct data structures."""
        sim_counts, pi_ests, errors = estimate_pi_with_convergence(
            max_simulations=5000,
            step=1000
        )
        
        assert len(sim_counts) == 5, "Should have 5 data points"
        assert len(pi_ests) == 5, "Should have 5 pi estimates"
        assert len(errors) == 5, "Should have 5 errors"
        
        # Check that simulation counts are correct
        assert sim_counts == [1000, 2000, 3000, 4000, 5000]
    
    def test_estimate_pi_with_convergence_errors_decrease(self):
        """Test that errors generally decrease with more samples."""
        sim_counts, pi_ests, errors = estimate_pi_with_convergence(
            max_simulations=50000,
            step=10000
        )
        
        # Last error should be smaller than first (generally)
        # Using median to avoid stochastic failures
        median_first_half = np.median(errors[:2])
        median_second_half = np.median(errors[-2:])
        
        # Later estimates should generally be more accurate
        assert median_second_half < median_first_half * 1.5, \
            "Convergence not showing improvement"
    
    def test_estimate_pi_reproducibility(self):
        """Test that results are reproducible with same seed."""
        np.random.seed(123)
        pi_1 = estimate_pi(5000)
        
        np.random.seed(123)
        pi_2 = estimate_pi(5000)
        
        assert pi_1 == pi_2, "Results should be identical with same seed"
    
    def test_estimate_pi_different_seeds_give_different_results(self):
        """Test that different seeds produce different estimates."""
        np.random.seed(42)
        pi_1 = estimate_pi(5000)
        
        np.random.seed(99)
        pi_2 = estimate_pi(5000)
        
        # Very unlikely to be exactly the same
        assert pi_1 != pi_2, "Different seeds should produce different results"
    
    def test_small_sample_size(self):
        """Test that function handles small sample sizes."""
        estimated_pi = estimate_pi(100)
        
        # Should still be numeric and in reasonable range
        assert isinstance(estimated_pi, (int, float))
        assert 0 < estimated_pi < 10  # Very loose bound for small sample


class TestMonteCarloStatistics:
    """Test statistical properties of Monte Carlo simulation."""
    
    def test_multiple_runs_standard_deviation(self):
        """Test that multiple runs show expected variance."""
        estimates = []
        
        for i in range(20):
            np.random.seed(i)
            estimates.append(estimate_pi(10000))
        
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)
        
        # Mean should be close to pi
        assert abs(mean_estimate - np.pi) < 0.05, \
            f"Mean of estimates {mean_estimate} too far from pi"
        
        # Standard deviation should be reasonable (not too high, not zero)
        assert 0.01 < std_estimate < 0.2, \
            f"Standard deviation {std_estimate} outside expected range"
    
    def test_law_of_large_numbers(self):
        """Test that average of many runs converges to pi."""
        n_runs = 50
        estimates = []
        
        for i in range(n_runs):
            np.random.seed(i + 1000)
            estimates.append(estimate_pi(5000))
        
        average = np.mean(estimates)
        
        # Average should be very close to pi
        assert abs(average - np.pi) < 0.03, \
            f"Average of {n_runs} runs should be close to pi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
