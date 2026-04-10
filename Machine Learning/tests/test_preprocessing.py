"""
Unit tests for preprocessing utilities
"""
import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.preprocessing import (
    standardize_features,
    normalize_features,
    encode_labels,
    add_polynomial_features
)
from utils.data_loader import load_iris_dataset


class TestStandardization(unittest.TestCase):
    """Test cases for feature standardization"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = load_iris_dataset()
    
    def test_standardize_output_shape(self):
        """Test that standardization preserves shape"""
        X_std, scaler = standardize_features(self.X)
        self.assertEqual(X_std.shape, self.X.shape,
                        "Standardized data should have same shape as input")
    
    def test_standardize_mean_zero(self):
        """Test that standardized features have mean ≈ 0"""
        X_std, scaler = standardize_features(self.X)
        
        means = X_std.mean(axis=0)
        for mean in means:
            self.assertAlmostEqual(mean, 0.0, places=10,
                                 msg="Standardized features should have mean ≈ 0")
    
    def test_standardize_std_one(self):
        """Test that standardized features have std ≈ 1"""
        X_std, scaler = standardize_features(self.X)
        
        stds = X_std.std(axis=0)
        for std in stds:
            self.assertAlmostEqual(std, 1.0, places=10,
                                 msg="Standardized features should have std ≈ 1")
    
    def test_standardize_inverse_transform(self):
        """Test that inverse transform recovers original data"""
        X_std, scaler = standardize_features(self.X)
        X_recovered = scaler.inverse_transform(X_std)
        
        np.testing.assert_array_almost_equal(
            X_recovered, self.X, decimal=10,
            err_msg="Inverse transform should recover original data"
        )


class TestNormalization(unittest.TestCase):
    """Test cases for feature normalization"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = load_iris_dataset()
    
    def test_normalize_output_shape(self):
        """Test that normalization preserves shape"""
        X_norm, scaler = normalize_features(self.X)
        self.assertEqual(X_norm.shape, self.X.shape,
                        "Normalized data should have same shape as input")
    
    def test_normalize_range(self):
        """Test that normalized features are in [0, 1]"""
        X_norm, scaler = normalize_features(self.X)
        
        self.assertGreaterEqual(X_norm.min(), 0.0,
                              "Normalized features should be >= 0")
        self.assertLessEqual(X_norm.max(), 1.0,
                           "Normalized features should be <= 1")
        
        # Check each feature has min=0 and max=1
        for i in range(X_norm.shape[1]):
            feature_min = X_norm[:, i].min()
            feature_max = X_norm[:, i].max()
            
            self.assertAlmostEqual(feature_min, 0.0, places=10,
                                 msg=f"Feature {i} should have min=0")
            self.assertAlmostEqual(feature_max, 1.0, places=10,
                                 msg=f"Feature {i} should have max=1")
    
    def test_normalize_inverse_transform(self):
        """Test that inverse transform recovers original data"""
        X_norm, scaler = normalize_features(self.X)
        X_recovered = scaler.inverse_transform(X_norm)
        
        np.testing.assert_array_almost_equal(
            X_recovered, self.X, decimal=10,
            err_msg="Inverse transform should recover original data"
        )


class TestLabelEncoding(unittest.TestCase):
    """Test cases for label encoding"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = load_iris_dataset()
    
    def test_encode_output_type(self):
        """Test that encoded labels are integers"""
        y_encoded, encoder = encode_labels(self.y)
        
        self.assertTrue(np.issubdtype(y_encoded.dtype, np.integer),
                       "Encoded labels should be integers")
    
    def test_encode_output_range(self):
        """Test that encoded labels are in correct range"""
        y_encoded, encoder = encode_labels(self.y)
        
        n_classes = len(np.unique(self.y))
        
        self.assertGreaterEqual(y_encoded.min(), 0,
                              "Encoded labels should start from 0")
        self.assertLess(y_encoded.max(), n_classes,
                       f"Encoded labels should be < {n_classes}")
    
    def test_encode_bijection(self):
        """Test that encoding is a bijection (one-to-one mapping)"""
        y_encoded, encoder = encode_labels(self.y)
        
        # Each unique original label should map to unique encoded label
        unique_original = np.unique(self.y)
        unique_encoded = np.unique(y_encoded)
        
        self.assertEqual(len(unique_original), len(unique_encoded),
                        "Number of unique labels should be preserved")
    
    def test_encode_inverse_transform(self):
        """Test that inverse transform recovers original labels"""
        y_encoded, encoder = encode_labels(self.y)
        y_recovered = encoder.inverse_transform(y_encoded)
        
        np.testing.assert_array_equal(
            y_recovered, self.y,
            err_msg="Inverse transform should recover original labels"
        )


class TestPolynomialFeatures(unittest.TestCase):
    """Test cases for polynomial feature generation"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = load_iris_dataset()
    
    def test_polynomial_degree_2(self):
        """Test polynomial features for degree 2"""
        X_poly, poly = add_polynomial_features(self.X, degree=2)
        
        # For 4 features with degree 2 and include_bias=False:
        # Features: [x1, x2, x3, x4, x1^2, x1*x2, x1*x3, x1*x4, x2^2, ...]
        # Total = 4 + C(4+2-1, 2) - 4 = 4 + 10 = 14
        expected_features = 14
        
        self.assertEqual(X_poly.shape[1], expected_features,
                        f"Degree 2 should produce {expected_features} features")
    
    def test_polynomial_includes_original(self):
        """Test that polynomial features include original features"""
        X_poly, poly = add_polynomial_features(self.X, degree=2)
        
        # First 4 features should be the original features
        np.testing.assert_array_almost_equal(
            X_poly[:, :4], self.X, decimal=10,
            err_msg="Original features should be included"
        )
    
    def test_polynomial_degree_1(self):
        """Test that degree 1 returns original features"""
        X_poly, poly = add_polynomial_features(self.X, degree=1)
        
        np.testing.assert_array_almost_equal(
            X_poly, self.X, decimal=10,
            err_msg="Degree 1 should return original features"
        )
    
    def test_polynomial_higher_degree(self):
        """Test polynomial features for higher degrees"""
        X_poly_2, _ = add_polynomial_features(self.X, degree=2)
        X_poly_3, _ = add_polynomial_features(self.X, degree=3)
        
        self.assertGreater(X_poly_3.shape[1], X_poly_2.shape[1],
                         "Higher degree should produce more features")


class TestPreprocessingWorkflow(unittest.TestCase):
    """Test complete preprocessing workflows"""
    
    def test_standardize_then_encode(self):
        """Test standardization followed by label encoding"""
        X, y = load_iris_dataset()
        
        # Standardize features
        X_std, scaler = standardize_features(X)
        
        # Encode labels
        y_enc, encoder = encode_labels(y)
        
        # Check both transformations worked
        self.assertEqual(X_std.shape, X.shape)
        self.assertEqual(y_enc.shape, y.shape)
        self.assertAlmostEqual(X_std.mean(), 0.0, places=5)
        self.assertTrue(np.issubdtype(y_enc.dtype, np.integer))
    
    def test_normalize_polynomial(self):
        """Test normalization with polynomial features"""
        X, y = load_iris_dataset()
        
        # Normalize
        X_norm, norm_scaler = normalize_features(X)
        
        # Add polynomial features
        X_poly, poly = add_polynomial_features(X_norm, degree=2)
        
        # Verify workflow
        self.assertGreaterEqual(X_norm.min(), 0.0)
        self.assertLessEqual(X_norm.max(), 1.0)
        self.assertEqual(X_poly.shape[0], X.shape[0])
        self.assertGreater(X_poly.shape[1], X.shape[1])


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestStandardization))
    suite.addTests(loader.loadTestsFromTestCase(TestNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestPolynomialFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessingWorkflow))
    
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
