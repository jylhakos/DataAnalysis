"""
LSTM Time-Series Forecasting for Weather Data

This module implements Long Short-Term Memory (LSTM) neural networks for
time-series forecasting, specifically targeting weather prediction tasks.
LSTMs are a specialized form of Recurrent Neural Networks (RNNs) designed
to capture long-term dependencies in sequential data.

Mathematical Foundation:
    
    RNN Hidden State Update:
    h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
    
    LSTM Gates:
    - Forget Gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    - Input Gate:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    - Output Gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
    - Cell State:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C · [h_{t-1}, x_t] + b_C)
    - Hidden State: h_t = o_t ⊙ tanh(C_t)
    
    Where:
    - σ is the sigmoid activation function
    - ⊙ denotes element-wise multiplication
    - W are weight matrices
    - b are bias vectors

References:
    Understanding LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    TensorFlow Time Series: https://www.tensorflow.org/tutorials/structured_data/time_series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Optional
import logging

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    SequentialType = Sequential
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")
    SequentialType = "Sequential"  # String type hint for when TensorFlow is not available


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RNNBasic:
    """
    Basic RNN implementation from scratch for educational purposes.
    Demonstrates the fundamental RNN forward pass calculation.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize basic RNN with random weights.
        
        Args:
            input_size: Dimensionality of input features
            hidden_size: Number of hidden state units
        """
        self.hidden_size = hidden_size
        # Xavier initialization for better gradient flow
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        
    def rnn_step(self, xt: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Single RNN forward step.
        
        Mathematical Formula:
            h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
        
        Args:
            xt: Input at time t, shape (input_size, 1)
            h_prev: Previous hidden state, shape (hidden_size, 1)
            
        Returns:
            h_next: New hidden state, shape (hidden_size, 1)
        """
        h_next = np.tanh(np.dot(self.Wxh, xt) + np.dot(self.Whh, h_prev) + self.bh)
        return h_next
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Process a sequence through the RNN.
        
        Args:
            inputs: Sequence of inputs, shape (sequence_length, input_size)
            
        Returns:
            outputs: Hidden states for each timestep
        """
        h = np.zeros((self.hidden_size, 1))
        outputs = []
        
        for xt in inputs:
            xt = xt.reshape(-1, 1)
            h = self.rnn_step(xt, h)
            outputs.append(h.copy())
            
        return np.array(outputs)


class WeatherTimeSeriesForecaster:
    """
    LSTM-based weather forecasting system for temperature prediction.
    
    This class implements a complete pipeline for time-series forecasting:
    1. Data loading and preprocessing
    2. Sequence creation with sliding window
    3. LSTM model construction
    4. Training with validation
    5. Prediction and evaluation
    """
    
    def __init__(self, 
                 sequence_length: int = 30,
                 prediction_horizon: int = 1,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2):
        """
        Initialize the forecaster.
        
        Args:
            sequence_length: Number of past timesteps to use for prediction (window size)
            prediction_horizon: Number of future timesteps to predict
            lstm_units: Number of LSTM units in each layer
            dropout_rate: Dropout rate for regularization (prevents overfitting)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
            
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model: Optional["Sequential"] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_names: list = []
        
        logger.info(f"Initialized forecaster: seq_len={sequence_length}, "
                   f"horizon={prediction_horizon}, units={lstm_units}")
    
    def load_weather_data(self, 
                         filepath: str,
                         target_column: str = 'T (degC)',
                         feature_columns: Optional[list] = None) -> pd.DataFrame:
        """
        Load and preprocess weather data from CSV file.
        
        Args:
            filepath: Path to CSV file (e.g., mpi_roof.csv)
            target_column: Column name for prediction target
            feature_columns: List of feature columns to use (None = all numeric)
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Loading weather data from {filepath}")
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Parse datetime
        if 'Date Time' in df.columns:
            df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
            df = df.set_index('Date Time')
        
        # Select features
        if feature_columns is None:
            # Use all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != target_column]
        
        # Ensure target is included
        if target_column not in feature_columns:
            feature_columns = [target_column] + feature_columns
        
        self.feature_names = feature_columns
        df_processed = df[feature_columns].copy()
        
        # Handle missing values
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Loaded {len(df_processed)} records with {len(feature_columns)} features")
        logger.info(f"Features: {feature_columns}")
        
        return df_processed
    
    def create_sequences(self, 
                        data: np.ndarray,
                        target_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for supervised learning.
        
        Transform time-series data into input-output pairs:
        X: [t-n, t-n+1, ..., t-1] → y: [t, t+1, ..., t+h-1]
        
        Args:
            data: Scaled feature array, shape (n_samples, n_features)
            target_index: Index of target feature in data
            
        Returns:
            X: Input sequences, shape (n_sequences, sequence_length, n_features)
            y: Target values, shape (n_sequences, prediction_horizon)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Input: past sequence_length timesteps
            X.append(data[i:(i + self.sequence_length), :])
            
            # Output: next prediction_horizon timesteps of target
            y.append(data[(i + self.sequence_length):
                         (i + self.sequence_length + self.prediction_horizon), 
                         target_index])
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        logger.info(f"Created sequences: X shape={X_array.shape}, y shape={y_array.shape}")
        return X_array, y_array
    
    def build_model(self, n_features: int) -> "Sequential":
        """
        Build LSTM neural network architecture.
        
        Architecture:
            1. LSTM Layer 1: Returns sequences for stacking
            2. Dropout Layer 1: Regularization
            3. LSTM Layer 2: Final sequence processing
            4. Dropout Layer 2: Regularization
            5. Dense Layer: Output prediction
        
        Args:
            n_features: Number of input features
            
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            # First LSTM layer - processes sequences and returns full sequence
            LSTM(units=self.lstm_units, 
                 activation='tanh',
                 return_sequences=True,
                 input_shape=(self.sequence_length, n_features),
                 name='lstm_layer_1'),
            
            # Dropout for regularization
            Dropout(rate=self.dropout_rate, name='dropout_1'),
            
            # Second LSTM layer - returns final hidden state only
            LSTM(units=self.lstm_units // 2,
                 activation='tanh',
                 return_sequences=False,
                 name='lstm_layer_2'),
            
            # Dropout for regularization
            Dropout(rate=self.dropout_rate, name='dropout_2'),
            
            # Output layer
            Dense(units=self.prediction_horizon, name='output_layer')
        ])
        
        # Compile with Adam optimizer and MSE loss
        model.compile(optimizer='adam', 
                     loss='mean_squared_error',
                     metrics=['mae'])
        
        logger.info("Model architecture built:")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 50,
             batch_size: int = 32,
             verbose: int = 1) -> dict:
        """
        Train LSTM model with early stopping.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Maximum number of training epochs
            batch_size: Mini-batch size for gradient descent
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting model training...")
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(n_features=X_train.shape[2])
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True,
                                   verbose=1)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        logger.info("Training completed")
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input sequences.
        
        Args:
            X: Input sequences, shape (n_sequences, sequence_length, n_features)
            
        Returns:
            Predictions, shape (n_sequences, prediction_horizon)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def evaluate(self, 
                y_true: np.ndarray,
                y_pred: np.ndarray) -> dict:
        """
        Calculate evaluation metrics for predictions.
        
        Metrics:
            - MSE: Mean Squared Error
            - RMSE: Root Mean Squared Error
            - MAE: Mean Absolute Error
            - R²: Coefficient of Determination
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        logger.info("Evaluation Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def plot_predictions(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        title: str = "Weather Forecast: Actual vs Predicted",
                        save_path: Optional[str] = None):
        """
        Visualize predictions against actual values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(14, 6))
        
        # For multi-step prediction, plot first step only
        if len(y_true.shape) > 1:
            y_true_plot = y_true[:, 0]
            y_pred_plot = y_pred[:, 0]
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
        
        plt.plot(y_true_plot, label='Actual', color='blue', alpha=0.7, linewidth=1.5)
        plt.plot(y_pred_plot, label='Predicted', color='red', alpha=0.7, linewidth=1.5)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()


def demonstrate_basic_rnn():
    """
    Demonstrate basic RNN forward pass with simple synthetic data.
    Educational function to show RNN computation.
    """
    print("\n" + "="*60)
    print("BASIC RNN DEMONSTRATION")
    print("="*60)
    
    # Initialize basic RNN
    rnn = RNNBasic(input_size=3, hidden_size=5)
    
    # Create synthetic sequence
    sequence = np.array([
        [0.5, 0.2, 0.8],
        [0.3, 0.6, 0.1],
        [0.9, 0.4, 0.7]
    ])
    
    print(f"\nInput sequence shape: {sequence.shape}")
    print(f"Input sequence:\n{sequence}\n")
    
    # Process sequence
    outputs = rnn.forward(sequence)
    
    print(f"Hidden states shape: {outputs.shape}")
    print(f"Final hidden state:\n{outputs[-1]}\n")
    print("="*60)


def main():
    """
    Main execution function demonstrating complete LSTM forecasting pipeline.
    """
    print("\n" + "="*80)
    print("LSTM TIME-SERIES WEATHER FORECASTING")
    print("="*80 + "\n")
    
    # Demonstrate basic RNN
    demonstrate_basic_rnn()
    
    # Initialize forecaster
    forecaster = WeatherTimeSeriesForecaster(
        sequence_length=24,  # Use 24 hours (2.4 hours per record * 10) of data
        prediction_horizon=1,  # Predict next timestep
        lstm_units=64,
        dropout_rate=0.2
    )
    
    # Load weather data
    data_path = "data/raw/mpi_roof.csv"
    
    try:
        df = forecaster.load_weather_data(
            filepath=data_path,
            target_column='T (degC)',
            feature_columns=['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']
        )
        
        # Scale data
        data_scaled = forecaster.scaler.fit_transform(df.values)
        
        # Create sequences
        X, y = forecaster.create_sequences(data_scaled, target_index=0)
        
        # Split data: 80% train, 20% test (maintaining temporal order)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Further split training into train and validation
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        print(f"\nDataset splits:")
        print(f"  Training: {len(X_train)} sequences")
        print(f"  Validation: {len(X_val)} sequences")
        print(f"  Testing: {len(X_test)} sequences\n")
        
        # Train model
        history = forecaster.train(
            X_train, y_train,
            X_val, y_val,
            epochs=50,
            batch_size=32,
            verbose=2
        )
        
        # Make predictions
        y_pred_train = forecaster.predict(X_train)
        y_pred_test = forecaster.predict(X_test)
        
        # Inverse transform predictions back to original scale
        # Create dummy arrays for inverse transform
        dummy_train = np.zeros((len(y_pred_train), df.shape[1]))
        dummy_test = np.zeros((len(y_pred_test), df.shape[1]))
        dummy_train[:, 0] = y_pred_train.flatten()
        dummy_test[:, 0] = y_pred_test.flatten()
        
        y_pred_train_orig = forecaster.scaler.inverse_transform(dummy_train)[:, 0]
        y_pred_test_orig = forecaster.scaler.inverse_transform(dummy_test)[:, 0]
        
        # Inverse transform actual values
        dummy_train_actual = np.zeros((len(y_train), df.shape[1]))
        dummy_test_actual = np.zeros((len(y_test), df.shape[1]))
        dummy_train_actual[:, 0] = y_train.flatten()
        dummy_test_actual[:, 0] = y_test.flatten()
        
        y_train_orig = forecaster.scaler.inverse_transform(dummy_train_actual)[:, 0]
        y_test_orig = forecaster.scaler.inverse_transform(dummy_test_actual)[:, 0]
        
        # Evaluate
        print("\n" + "="*60)
        print("TRAINING SET METRICS")
        print("="*60)
        train_metrics = forecaster.evaluate(y_train_orig, y_pred_train_orig)
        
        print("\n" + "="*60)
        print("TEST SET METRICS")
        print("="*60)
        test_metrics = forecaster.evaluate(y_test_orig, y_pred_test_orig)
        
        # Plot results
        forecaster.plot_predictions(
            y_test_orig[:200],  # Plot first 200 predictions
            y_pred_test_orig[:200],
            title="Temperature Forecast: Actual vs Predicted (Test Set)",
            save_path="notebooks/weather_forecast_results.png"
        )
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please ensure mpi_roof.csv is in data/raw/ directory")
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
