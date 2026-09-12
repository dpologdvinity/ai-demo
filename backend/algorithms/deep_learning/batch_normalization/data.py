"""Data generation for Batch Normalization demonstration."""

from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import TensorDataset


def generate_training_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 10,
    noise_level: float = 0.1,
    random_state: int = 42
) -> Dict[str, Any]:
    """Generate synthetic classification data for BN demonstration.

    Creates a multi-class classification dataset with features that have
    varying scales and distributions, which helps demonstrate the
    effectiveness of batch normalization.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        n_classes: Number of output classes
        noise_level: Amount of noise to add to labels
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - X_train: Training features (N, n_features)
            - X_test: Test features (M, n_features)
            - y_train: Training labels (N,)
            - y_test: Test labels (M,)
            - train_dataset: PyTorch TensorDataset for training
            - test_dataset: PyTorch TensorDataset for testing
            - n_features: Number of features
            - n_classes: Number of classes
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Generate features with different scales to show BN benefits
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        # Vary the scale across features (some large, some small)
        scale = 10 ** np.random.uniform(-1, 2)
        X[:, i] = np.random.randn(n_samples) * scale

    # Generate labels based on linear combination of features
    weights = np.random.randn(n_features, n_classes)
    logits = X @ weights

    # Add noise
    logits += np.random.randn(n_samples, n_classes) * noise_level

    # Convert to class labels
    y = np.argmax(logits, axis=1)

    # Split into train and test
    split_idx = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = X[train_indices].astype(np.float32)
    X_test = X[test_indices].astype(np.float32)
    y_train = y[train_indices]
    y_test = y[test_indices]

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test).long()

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'n_features': n_features,
        'n_classes': n_classes
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the synthetic dataset used for BN demonstration.

    Returns:
        Dictionary with dataset metadata
    """
    return {
        'name': 'Synthetic Classification Data',
        'description': 'Multi-class classification with varying feature scales',
        'num_samples': 1000,
        'num_features': 20,
        'num_classes': 10,
        'train_split': 0.8,
        'test_split': 0.2,
        'characteristics': [
            'Features have different scales (demonstrating BN normalization)',
            'Non-linear class boundaries',
            'Controlled noise level for reproducibility',
            'Balanced class distribution'
        ],
        'use_case': 'Demonstrating batch normalization benefits on deep networks'
    }
