"""Data generation and utilities for Dropout demonstration."""

import numpy as np
from typing import Dict, Any
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def generate_overfitting_prone_dataset(
    n_samples: int = 200,
    n_features: int = 20,
    n_informative: int = 10,
    n_classes: int = 2,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    """Generate a small dataset that's prone to overfitting.

    This creates a synthetic dataset with limited samples but many features,
    making it easy to demonstrate the overfitting problem and how dropout helps.

    Args:
        n_samples: Total number of samples (kept small to encourage overfitting)
        n_features: Number of input features
        n_informative: Number of informative features
        n_classes: Number of output classes
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
            - X_val: Validation features
            - y_val: Validation labels
            - n_features: Number of features
            - n_classes: Number of classes
    """
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_classes=n_classes,
        n_clusters_per_class=2,
        flip_y=0.1,  # Add some noise
        class_sep=0.8,  # Not too easy to separate
        random_state=random_state
    )

    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Split into train and temp (test + val)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split temp into test and validation
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val,
        'n_features': n_features,
        'n_classes': n_classes
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the dropout demonstration dataset.

    Returns:
        Dictionary with dataset metadata
    """
    return {
        'name': 'Overfitting-Prone Synthetic Dataset',
        'type': 'classification',
        'n_samples': 200,
        'n_features': 20,
        'n_informative_features': 10,
        'n_classes': 2,
        'purpose': 'Demonstrate overfitting and dropout regularization',
        'description': (
            'A small synthetic binary classification dataset designed to be prone to overfitting. '
            'The limited number of samples (200) compared to features (20) makes it easy for '
            'neural networks to memorize the training data. This demonstrates how dropout '
            'prevents overfitting by randomly dropping neurons during training, forcing the '
            'network to learn more robust features.'
        ),
        'train_size': 140,
        'validation_size': 30,
        'test_size': 30
    }


def create_dropout_mask(shape: tuple, dropout_rate: float, random_state: int = None) -> np.ndarray:
    """Create a binary dropout mask.

    During training, this mask is used to randomly set neurons to zero.
    During inference, all neurons are active (no dropout).

    Args:
        shape: Shape of the layer (e.g., (batch_size, n_neurons))
        dropout_rate: Probability of dropping each neuron
        random_state: Random seed

    Returns:
        Binary mask where 1 = keep neuron, 0 = drop neuron
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random

    # Create mask: 1 to keep, 0 to drop
    keep_prob = 1.0 - dropout_rate
    mask = rng.binomial(1, keep_prob, size=shape)

    # Scale by 1/keep_prob to maintain expected value (inverted dropout)
    if keep_prob > 0:
        mask = mask / keep_prob

    return mask.astype(np.float32)
