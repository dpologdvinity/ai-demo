"""Data loading utilities for VAE training."""

from typing import Dict, Any
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_vae_data(
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True
) -> Dict[str, Any]:
    """Load and prepare digits dataset for VAE training.

    Loads the 8x8 digits dataset from scikit-learn and prepares it
    for VAE training by splitting into train/test sets and optionally
    normalizing.

    Args:
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility
        normalize: Whether to normalize the data to [0, 1] range

    Returns:
        Dictionary containing:
            - X_train: Training images, shape (n_train, 64)
            - X_test: Test images, shape (n_test, 64)
            - y_train: Training labels
            - y_test: Test labels
            - input_dim: Input dimension (64 for 8x8 images)
            - n_samples: Total number of samples
            - n_features: Number of features (64)
            - image_shape: Shape of images (8, 8)
    """
    # Load the digits dataset
    digits = load_digits()
    X = digits.data  # Already flattened 8x8 images (64 features)
    y = digits.target

    # Normalize to [0, 1] range if requested
    if normalize:
        X = X / 16.0  # Digits are in range [0, 16]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'input_dim': X.shape[1],
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'image_shape': (8, 8)
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the digits dataset.

    Returns:
        Dictionary with dataset information
    """
    digits = load_digits()

    return {
        "name": "Digits",
        "description": "8x8 grayscale images of handwritten digits (0-9)",
        "n_samples": digits.data.shape[0],
        "n_features": digits.data.shape[1],
        "n_classes": len(np.unique(digits.target)),
        "image_shape": [8, 8],
        "classes": list(range(10)),
        "feature_range": [0, 16],
        "normalized_range": [0, 1]
    }
