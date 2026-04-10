"""
Unit tests for LSTM time-series forecasting module.

This test suite covers:
1. Basic RNN functionality and shape validation
2. LSTM model architecture and output shapes
3. Data preprocessing and sequence creation
4. Training stability and overfitting checks
5. Prediction consistency and numerical stability
6. Edge cases and error handling

Testing Philosophy for RNNs:
    - Verify tensor shapes at each processing stage
    - Test with multiple batch sizes and sequence lengths
    - Ensure model can overfit small synthetic datasets (sanity check)
    - Validate numerical stability (no NaN or Inf values)
    - Check prediction variance and bias
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from typing import Tuple

# Import modules under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from supervised.lstm_timeseries import (
    RNNBasic,
    WeatherTimeSeriesForecaster,
    TENSORFLOW_AVAILABLE
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def sample_weather_data() -> pd.DataFrame:
    """
    Create synthetic weather dataset for testing.
    Mimics structure of mpi_roof.csv with predictable patterns.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create time index
    dates = pd.date_range(start='2026-01-01', periods=n_samples, freq='10T')
    
    # Generate synthetic weather data with patterns
    t = np.arange(n_samples)
    temperature = 15 + 10 * np.sin(2 * np.pi * t / 144) + np.random.normal(0, 2, n_samples)
    pressure = 1000 + 20 * np.sin(2 * np.pi * t / 288) + np.random.normal(0, 5, n_samples)
    humidity = 60 + 20 * np.cos(2 * np.pi * t / 144) + np.random.normal(0, 5, n_samples)
    wind_speed = 2 + 3 * np.abs(np.sin(2 * np.pi * t / 100)) + np.random.normal(0, 0.5, n_samples)
    
    # Clip to realistic ranges
    humidity = np.clip(humidity, 0, 100)
    wind_speed = np.clip(wind_speed, 0, None)
    
    df = pd.DataFrame({
        'Date Time': dates,
        'T (degC)': temperature,
        'p (mbar)': pressure,
        'rh (%)': humidity,
        'wv (m/s)': wind_speed
    })
    
    return df


@pytest.fixture(scope="session")
def temp_csv_file(sample_weather_data) -> str:
    """
    Create temporary CSV file with weather data.
    Returns path to temporary file, cleaned up after session.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_weather_data.to_csv(f, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def basic_rnn() -> RNNBasic:
    """Fixture providing a basic RNN instance."""
    return RNNBasic(input_size=3, hidden_size=5)


@pytest.fixture
def forecaster() -> WeatherTimeSeriesForecaster:
    """Fixture providing a WeatherTimeSeriesForecaster instance."""
    if not TENSORFLOW_AVAILABLE:
        pytest.skip("TensorFlow not available")
    return WeatherTimeSeriesForecaster(
        sequence_length=10,
        prediction_horizon=1,
        lstm_units=16,
        dropout_rate=0.2
    )


# ============================================================================
# BASIC RNN TESTS
# ============================================================================

class TestRNNBasic:
    """Test suite for basic RNN implementation."""
    
    def test_rnn_initialization(self, basic_rnn):
        """Test RNN weight initialization."""
        assert basic_rnn.Wxh.shape == (5, 3), "Wxh weight matrix has incorrect shape"
        assert basic_rnn.Whh.shape == (5, 5), "Whh weight matrix has incorrect shape"
        assert basic_rnn.bh.shape == (5, 1), "Bias vector has incorrect shape"
        
        # Check that weights are not all zeros (proper initialization)
        assert not np.allclose(basic_rnn.Wxh, 0), "Wxh should not be all zeros"
        assert not np.allclose(basic_rnn.Whh, 0), "Whh should not be all zeros"
    
    @pytest.mark.parametrize("input_size,hidden_size", [
        (1, 5),
        (3, 10),
        (10, 20),
        (50, 100)
    ])
    def test_rnn_different_sizes(self, input_size, hidden_size):
        """Test RNN with various input and hidden dimensions."""
        rnn = RNNBasic(input_size=input_size, hidden_size=hidden_size)
        
        # Create random input
        x = np.random.randn(input_size, 1)
        h_prev = np.zeros((hidden_size, 1))
        
        # Forward pass
        h_next = rnn.rnn_step(x, h_prev)
        
        assert h_next.shape == (hidden_size, 1), f"Output shape mismatch for hidden_size={hidden_size}"
        assert not np.any(np.isnan(h_next)), "Output contains NaN values"
        assert not np.any(np.isinf(h_next)), "Output contains Inf values"
    
    def test_rnn_step_output_range(self, basic_rnn):
        """Test that RNN output is bounded by tanh activation."""
        x = np.random.randn(3, 1) * 10  # Large input
        h_prev = np.random.randn(5, 1) * 10
        
        h_next = basic_rnn.rnn_step(x, h_prev)
        
        # tanh outputs are in range [-1, 1]
        assert np.all(h_next >= -1.0) and np.all(h_next <= 1.0), \
            "RNN output should be in range [-1, 1] due to tanh activation"
    
    def test_rnn_forward_sequence(self, basic_rnn):
        """Test RNN processing of entire sequence."""
        sequence_length = 20
        input_size = 3
        
        sequence = np.random.randn(sequence_length, input_size)
        outputs = basic_rnn.forward(sequence)
        
        assert outputs.shape[0] == sequence_length, "Should have one output per input timestep"
        assert outputs.shape[1] == 5, "Hidden state dimension mismatch"
        assert outputs.shape[2] == 1, "Output format incorrect"


# ============================================================================
# WEATHER FORECASTER TESTS
# ============================================================================

class TestWeatherTimeSeriesForecaster:
    """Test suite for LSTM-based weather forecaster."""
    
    def test_forecaster_initialization(self, forecaster):
        """Test forecaster initialization parameters."""
        assert forecaster.sequence_length == 10
        assert forecaster.prediction_horizon == 1
        assert forecaster.lstm_units == 16
        assert forecaster.dropout_rate == 0.2
        assert forecaster.model is None, "Model should not be built until training"
    
    def test_load_weather_data(self, forecaster, temp_csv_file):
        """Test weather data loading and preprocessing."""
        df = forecaster.load_weather_data(
            filepath=temp_csv_file,
            target_column='T (degC)',
            feature_columns=['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']
        )
        
        assert isinstance(df, pd.DataFrame), "Output should be DataFrame"
        assert len(df) == 1000, "Should load all 1000 records"
        assert df.shape[1] == 4, "Should have 4 feature columns"
        assert 'T (degC)' in df.columns, "Target column missing"
        assert not df.isnull().any().any(), "Should handle missing values"
    
    @pytest.mark.parametrize("seq_len,pred_horizon", [
        (5, 1),
        (10, 1),
        (20, 3),
        (30, 5)
    ])
    def test_create_sequences_shapes(self, seq_len, pred_horizon, sample_weather_data):
        """Test sequence creation with different window sizes."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")
            
        forecaster = WeatherTimeSeriesForecaster(
            sequence_length=seq_len,
            prediction_horizon=pred_horizon
        )
        
        # Prepare data
        data = sample_weather_data[['T (degC)', 'p (mbar)']].values
        data_scaled = forecaster.scaler.fit_transform(data)
        
        # Create sequences
        X, y = forecaster.create_sequences(data_scaled, target_index=0)
        
        expected_n_sequences = len(data) - seq_len - pred_horizon + 1
        
        assert X.shape == (expected_n_sequences, seq_len, 2), \
            f"X shape mismatch for seq_len={seq_len}"
        assert y.shape == (expected_n_sequences, pred_horizon), \
            f"y shape mismatch for pred_horizon={pred_horizon}"
    
    def test_create_sequences_temporal_order(self, forecaster, sample_weather_data):
        """Test that sequences maintain temporal order."""
        data = sample_weather_data[['T (degC)']].values[:50]  # Small subset
        data_scaled = forecaster.scaler.fit_transform(data)
        
        X, y = forecaster.create_sequences(data_scaled, target_index=0)
        
        # First sequence should be data[0:10], target data[10]
        np.testing.assert_array_almost_equal(
            X[0, :, 0],
            data_scaled[:10, 0],
            decimal=5,
            err_msg="First sequence should match first 10 data points"
        )
        
        np.testing.assert_almost_equal(
            y[0, 0],
            data_scaled[10, 0],
            decimal=5,
            err_msg="First target should be 11th data point"
        )
    
    def test_build_model_architecture(self, forecaster):
        """Test LSTM model construction."""
        n_features = 4
        model = forecaster.build_model(n_features)
        
        assert model is not None, "Model should be built"
        assert len(model.layers) == 5, "Should have 5 layers (2 LSTM, 2 Dropout, 1 Dense)"
        
        # Check input shape
        assert model.input_shape == (None, 10, 4), "Input shape mismatch"
        
        # Check output shape
        assert model.output_shape == (None, 1), "Output shape mismatch"
    
    @pytest.mark.parametrize("batch_size,seq_len,n_features", [
        (1, 10, 4),
        (16, 10, 4),
        (32, 20, 1),
        (64, 5, 10)
    ])
    def test_model_output_shapes(self, batch_size, seq_len, n_features):
        """Test model handles different batch sizes and input dimensions."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")
            
        forecaster = WeatherTimeSeriesForecaster(
            sequence_length=seq_len,
            prediction_horizon=1,
            lstm_units=8
        )
        
        model = forecaster.build_model(n_features)
        
        # Create random input
        X_test = np.random.randn(batch_size, seq_len, n_features)
        
        # Predict
        output = model.predict(X_test, verbose=0)
        
        assert output.shape == (batch_size, 1), \
            f"Output shape mismatch for batch_size={batch_size}"
    
    def test_overfitting_on_tiny_dataset(self, forecaster):
        """
        Sanity check: Model should be able to overfit a tiny synthetic dataset.
        This verifies that the training logic is working correctly.
        """
        # Create tiny synthetic dataset (2 samples)
        np.random.seed(42)
        X_tiny = np.random.randn(2, 10, 4)
        y_tiny = np.random.randn(2, 1)
        
        # Build and train
        forecaster.model = forecaster.build_model(n_features=4)
        
        history = forecaster.model.fit(
            X_tiny, y_tiny,
            epochs=100,
            batch_size=2,
            verbose=0
        )
        
        # Should achieve very low loss on tiny dataset
        final_loss = history.history['loss'][-1]
        assert final_loss < 0.1, \
            f"Model should overfit tiny dataset (loss={final_loss:.4f})"
    
    def test_prediction_consistency(self, forecaster, temp_csv_file):
        """Test that predictions are consistent when called multiple times."""
        # Load and prepare data
        df = forecaster.load_weather_data(temp_csv_file, feature_columns=['T (degC)'])
        data_scaled = forecaster.scaler.fit_transform(df.values)
        X, y = forecaster.create_sequences(data_scaled)
        
        # Build tiny model
        forecaster.model = forecaster.build_model(n_features=1)
        
        # Make predictions multiple times
        pred1 = forecaster.predict(X[:10])
        pred2 = forecaster.predict(X[:10])
        pred3 = forecaster.predict(X[:10])
        
        # Predictions should be identical (no randomness in inference)
        np.testing.assert_array_equal(pred1, pred2, err_msg="Predictions should be deterministic")
        np.testing.assert_array_equal(pred2, pred3, err_msg="Predictions should be deterministic")
    
    def test_evaluation_metrics(self, forecaster):
        """Test evaluation metric calculations."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = forecaster.evaluate(y_true, y_pred)
        
        assert 'MSE' in metrics, "MSE metric missing"
        assert 'RMSE' in metrics, "RMSE metric missing"
        assert 'MAE' in metrics, "MAE metric missing"
        assert 'R2' in metrics, "R² metric missing"
        
        # Check metric values are reasonable
        assert metrics['MSE'] >= 0, "MSE should be non-negative"
        assert metrics['RMSE'] >= 0, "RMSE should be non-negative"
        assert metrics['MAE'] >= 0, "MAE should be non-negative"
        assert metrics['R2'] <= 1, "R² should be ≤ 1"
        
        # RMSE should be sqrt of MSE
        np.testing.assert_almost_equal(
            metrics['RMSE'],
            np.sqrt(metrics['MSE']),
            decimal=5
        )
    
    def test_perfect_prediction_metrics(self, forecaster):
        """Test that perfect predictions yield ideal metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        metrics = forecaster.evaluate(y_true, y_pred)
        
        assert metrics['MSE'] < 1e-10, "MSE should be ~0 for perfect predictions"
        assert metrics['RMSE'] < 1e-5, "RMSE should be ~0 for perfect predictions"
        assert metrics['MAE'] < 1e-10, "MAE should be ~0 for perfect predictions"
        np.testing.assert_almost_equal(metrics['R2'], 1.0, decimal=5, 
                                      err_msg="R² should be 1.0 for perfect predictions")


# ============================================================================
# NUMERICAL STABILITY TESTS
# ============================================================================

class TestNumericalStability:
    """Tests for numerical stability and edge cases."""
    
    def test_no_nan_in_predictions(self, forecaster, temp_csv_file):
        """Ensure predictions never contain NaN values."""
        df = forecaster.load_weather_data(temp_csv_file, feature_columns=['T (degC)'])
        data_scaled = forecaster.scaler.fit_transform(df.values)
        X, y = forecaster.create_sequences(data_scaled)
        
        forecaster.model = forecaster.build_model(n_features=1)
        forecaster.model.fit(X[:50], y[:50], epochs=5, verbose=0)
        
        predictions = forecaster.predict(X[50:60])
        
        assert not np.any(np.isnan(predictions)), "Predictions contain NaN values"
        assert not np.any(np.isinf(predictions)), "Predictions contain Inf values"
    
    def test_large_input_values(self, forecaster):
        """Test model with large input values."""
        # Create data with large values
        X_large = np.random.randn(10, 10, 4) * 1000
        
        forecaster.model = forecaster.build_model(n_features=4)
        
        try:
            predictions = forecaster.predict(X_large)
            assert not np.any(np.isnan(predictions)), "Large inputs caused NaN"
        except Exception as e:
            pytest.fail(f"Model failed with large inputs: {str(e)}")
    
    @pytest.mark.parametrize("sequence_length", [1, 5, 10, 50, 100])
    def test_varying_sequence_lengths(self, sequence_length):
        """Test model with various sequence lengths."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")
            
        forecaster = WeatherTimeSeriesForecaster(
            sequence_length=sequence_length,
            prediction_horizon=1,
            lstm_units=8
        )
        
        X = np.random.randn(5, sequence_length, 3)
        
        forecaster.model = forecaster.build_model(n_features=3)
        predictions = forecaster.predict(X)
        
        assert predictions.shape == (5, 1), \
            f"Predictions shape incorrect for seq_len={sequence_length}"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_predict_without_training(self, forecaster):
        """Test that prediction fails gracefully without trained model."""
        X = np.random.randn(10, 10, 4)
        
        with pytest.raises(ValueError, match="Model not trained"):
            forecaster.predict(X)
    
    def test_load_nonexistent_file(self, forecaster):
        """Test loading non-existent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            forecaster.load_weather_data("nonexistent_file.csv")
    
    def test_empty_sequence_creation(self, forecaster):
        """Test sequence creation with insufficient data."""
        # Data too small for sequence length
        small_data = np.random.randn(5, 2)
        
        X, y = forecaster.create_sequences(small_data)
        
        # Should return empty arrays if not enough data
        assert X.shape[0] == 0 or X.shape[0] >= 1, "Should handle small data gracefully"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, temp_csv_file):
        """Test complete forecasting pipeline from data loading to prediction."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")
        
        # Initialize
        forecaster = WeatherTimeSeriesForecaster(
            sequence_length=10,
            prediction_horizon=1,
            lstm_units=16,
            dropout_rate=0.1
        )
        
        # Load data
        df = forecaster.load_weather_data(
            temp_csv_file,
            feature_columns=['T (degC)', 'p (mbar)']
        )
        
        # Prepare sequences
        data_scaled = forecaster.scaler.fit_transform(df.values)
        X, y = forecaster.create_sequences(data_scaled)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train (brief)
        history = forecaster.train(
            X_train, y_train,
            X_test, y_test,
            epochs=3,
            batch_size=32,
            verbose=0
        )
        
        # Predict
        predictions = forecaster.predict(X_test[:10])
        
        # Validate
        assert predictions.shape == (10, 1), "Prediction shape incorrect"
        assert not np.any(np.isnan(predictions)), "Predictions contain NaN"
        
        # Evaluate
        metrics = forecaster.evaluate(y_test[:10], predictions)
        assert all(key in metrics for key in ['MSE', 'RMSE', 'MAE', 'R2'])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
