"""Data generation utilities for RNN time series prediction.

This module provides functions to generate synthetic sine wave data with noise
for RNN training and evaluation.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


def generate_sine_wave_data(
    n_samples: int = 1000,
    sequence_length: int = 20,
    prediction_length: int = 10,
    noise_level: float = 0.1,
    random_state: int = 42
) -> Dict[str, Any]:
    """Generate synthetic sine wave time series data with noise.

    Creates a sine wave with added Gaussian noise, suitable for training
    RNN models on sequence prediction tasks.

    Args:
        n_samples: Total number of time steps to generate
        sequence_length: Length of input sequences
        prediction_length: Length of sequences to predict
        noise_level: Standard deviation of Gaussian noise (0 = no noise)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - data: Complete time series (numpy array)
            - sequences: Input sequences for training
            - targets: Target sequences to predict
            - train_indices: Indices for training data
            - test_indices: Indices for test data
            - time_steps: Time values for plotting

    Example:
        >>> data = generate_sine_wave_data(
        ...     n_samples=1000,
        ...     sequence_length=20,
        ...     prediction_length=10,
        ...     noise_level=0.1
        ... )
    """
    np.random.seed(random_state)

    # Generate time steps
    time_steps = np.linspace(0, 4 * np.pi, n_samples)

    # Generate sine wave with noise
    sine_wave = np.sin(time_steps)
    noise = np.random.normal(0, noise_level, n_samples)
    data = sine_wave + noise

    # Create sequences
    sequences = []
    targets = []

    total_length = sequence_length + prediction_length
    for i in range(len(data) - total_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length:i + total_length]
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    # Split into train/test (80/20)
    split_idx = int(len(sequences) * 0.8)
    train_indices = list(range(split_idx))
    test_indices = list(range(split_idx, len(sequences)))

    return {
        "data": data,
        "sequences": sequences,
        "targets": targets,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "time_steps": time_steps
    }


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences.

    Attributes:
        sequences: Input sequences
        targets: Target sequences to predict
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """Initialize dataset.

        Args:
            sequences: Input sequences, shape (n_samples, sequence_length)
            targets: Target sequences, shape (n_samples, prediction_length)
        """
        self.sequences = torch.FloatTensor(sequences).unsqueeze(-1)  # Add feature dimension
        self.targets = torch.FloatTensor(targets).unsqueeze(-1)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (sequence, target)
        """
        return self.sequences[idx], self.targets[idx]


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the synthetic sine wave dataset.

    Returns:
        Dictionary containing dataset metadata
    """
    return {
        "name": "Synthetic Sine Wave",
        "description": "Time series data generated from a sine wave with Gaussian noise",
        "type": "Regression (Time Series)",
        "features": 1,
        "temporal": True,
        "source": "Synthetic"
    }
