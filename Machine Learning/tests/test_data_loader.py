"""
Unit tests for data loading utilities
"""
import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.data_loader import load_iris_dataset, split_data


class TestDataLoader(unittest.TestCase):
    """Test cases for data_loader module"""
    
    def test_load_iris_dataset(self):
        """Test loading Iris dataset"""
        X, y = load_iris_dataset()
        
        # Check shapes
        self.assertEqual(X.shape, (150, 4), "Features shape should be (150, 4)")
        self.assertEqual(y.shape, (150,), "Labels shape should be (150,)")
        
        # Check data types
        self.assertIsInstance(X, np.ndarray, "Features should be numpy array")
        self.assertIsInstance(y, np.ndarray, "Labels should be numpy array")
        
        # Check no missing values
        self.assertFalse(np.isnan(X).any(), "Features should not contain NaN")
        self.assertFalse(pd.isna(y).any(), "Labels should not contain NaN")
        
        # Check feature ranges
        self.assertGreater(X.min(), 0, "All features should be positive")
        self.assertLess(X.max(), 10, "Feature values should be reasonable")
        
        # Check number of classes
        unique_classes = np.unique(y)
        self.assertEqual(len(unique_classes), 3, "Should have 3 species")
    
    def test_split_data(self):
        """Test train-test split"""
        X, y = load_iris_dataset()
        
        # Test default split (70-30)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
        
        # Check sizes
        total_size = len(X)
        expected_train = int(total_size * 0.7)
        expected_test = total_size - expected_train
        
        self.assertEqual(len(X_train), expected_train, 
                        f"Training set should have ~{expected_train} samples")
        self.assertEqual(len(X_test), expected_test,
                        f"Test set should have ~{expected_test} samples")
        
        # Check shapes match
        self.assertEqual(len(X_train), len(y_train), 
                        "X_train and y_train should have same length")
        self.assertEqual(len(X_test), len(y_test),
                        "X_test and y_test should have same length")
        
        # Check feature dimensions preserved
        self.assertEqual(X_train.shape[1], X.shape[1],
                        "Feature dimensions should be preserved")
    
    def test_split_data_reproducibility(self):
        """Test that split is reproducible with same random_state"""
        X, y = load_iris_dataset()
        
        # Split with same random state
        X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=42)
        
        # Check reproducibility
        np.testing.assert_array_equal(X_train1, X_train2,
                                     "Splits should be identical with same random_state")
        np.testing.assert_array_equal(y_train1, y_train2,
                                     "Label splits should be identical")
    
    def test_split_data_stratification(self):
        """Test that stratified split maintains class proportions"""
        X, y = load_iris_dataset()
        
        # Get original class proportions
        unique, counts = np.unique(y, return_counts=True)
        original_proportions = counts / len(y)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
        
        # Check train set proportions
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        train_proportions = train_counts / len(y_train)
        
        # Test set proportions
        test_unique, test_counts = np.unique(y_test, return_counts=True)
        test_proportions = test_counts / len(y_test)
        
        # Proportions should be similar (within 10%)
        for orig_prop, train_prop, test_prop in zip(original_proportions, 
                                                     train_proportions, 
                                                     test_proportions):
            self.assertAlmostEqual(orig_prop, train_prop, delta=0.1,
                                 msg="Train proportions should match original")
            self.assertAlmostEqual(orig_prop, test_prop, delta=0.1,
                                 msg="Test proportions should match original")


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and quality"""
    
    def test_feature_correlations(self):
        """Test that features have expected correlations"""
        X, y = load_iris_dataset()
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(X.T)
        
        # Petal length and petal width should be highly correlated
        petal_corr = correlation_matrix[2, 3]
        self.assertGreater(petal_corr, 0.9,
                         "Petal length and width should be highly correlated")
    
    def test_class_separability(self):
        """Test that classes are reasonably separable"""
        X, y = load_iris_dataset()
        
        # Calculate mean feature values for each class
        classes = np.unique(y)
        means = []
        
        for cls in classes:
            mask = (y == cls)
            class_mean = X[mask].mean(axis=0)
            means.append(class_mean)
        
        means = np.array(means)
        
        # Check that class means are different
        for i in range(len(means)):
            for j in range(i+1, len(means)):
                distance = np.linalg.norm(means[i] - means[j])
                self.assertGreater(distance, 1.0,
                                 f"Classes {i} and {j} should have distinct means")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*50)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
