"""Data generation and preprocessing for GRU algorithm."""

import numpy as np
from typing import Dict, Any, Tuple
import torch


def generate_time_series(
    n_samples: int = 1000,
    sequence_length: int = 20,
    random_state: int = 42
) -> Dict[str, Any]:
    """Generate synthetic time series for GRU training.

    Creates a time series based on sine waves with noise that requires
    the GRU's gating mechanisms to learn patterns effectively.

    Args:
        n_samples: Total number of time steps to generate
        sequence_length: Length of input sequences
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - data: Full time series array
            - X: Input sequences (n_sequences, sequence_length)
            - y: Target values (n_sequences,)
            - X_train: Training input sequences
            - y_train: Training targets
            - X_test: Test input sequences
            - y_test: Test targets
            - info: Dataset metadata
    """
    np.random.seed(random_state)

    # Generate time index
    t = np.linspace(0, 100, n_samples)

    # Create complex sine wave pattern with multiple frequencies
    # Primary sine wave
    primary = 3 * np.sin(2 * np.pi * t / 20)

    # Secondary sine wave (different frequency)
    secondary = 1.5 * np.sin(2 * np.pi * t / 7)

    # Tertiary sine wave (slow frequency)
    tertiary = 0.5 * np.sin(2 * np.pi * t / 40)

    # Add some noise
    noise = np.random.normal(0, 0.2, n_samples)

    # Combine all components
    data = primary + secondary + tertiary + noise

    # Create sequences
    X_list = []
    y_list = []

    for i in range(len(data) - sequence_length):
        X_list.append(data[i:i + sequence_length])
        y_list.append(data[i + sequence_length])

    X = np.array(X_list)
    y = np.array(y_list)

    # Reshape for GRU: (n_sequences, sequence_length, 1)
    X = X.reshape(-1, sequence_length, 1)
    y = y.reshape(-1, 1)

    # Split into train/test (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return {
        'data': data,
        'X': X,
        'y': y,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'info': {
            'n_samples': n_samples,
            'sequence_length': sequence_length,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'description': 'Synthetic time series with multiple sine wave components'
        }
    }


def prepare_sequences(
    X: np.ndarray,
    y: np.ndarray,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert numpy arrays to PyTorch tensors.

    Args:
        X: Input sequences (n_sequences, sequence_length, n_features)
        y: Target values (n_sequences, 1)
        device: Device to place tensors on ('cpu' or 'cuda')

    Returns:
        Tuple of (X_tensor, y_tensor)
    """
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)

    return X_tensor, y_tensor


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the GRU dataset.

    Returns:
        Dictionary with dataset metadata
    """
    return {
        'name': 'Sine Wave Time Series',
        'description': 'Time series with multiple sine wave frequencies for sequence prediction',
        'type': 'Synthetic',
        'features': ['Time series values (sine waves)'],
        'target': 'Next time step value',
        'properties': [
            'Multiple frequency components',
            'Requires gating to capture patterns',
            'Suitable for testing GRU sequence modeling',
            'Contains both short-term and longer-term patterns'
        ]
    }
