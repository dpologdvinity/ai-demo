"""
Data loading and preparation for K-Nearest Neighbors algorithm.

This module handles loading the Iris dataset and preparing it for KNN
classification, including train/test splitting and feature scaling.
"""

from typing import Dict, Tuple, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.datasets import DatasetManager


def load_and_prepare_data(
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    """Load and prepare the Iris dataset for KNN classification.

    The Iris dataset is loaded from DatasetManager, which provides 150 samples
    with 4 features (sepal length, sepal width, petal length, petal width) and
    3 target classes (setosa, versicolor, virginica).

    Args:
        test_size: Proportion of data to use for testing (0.1 to 0.5)
        random_state: Random seed for reproducible splits

    Returns:
        Dictionary containing:
            - X_train: Training features (scaled)
            - X_test: Test features (scaled)
            - y_train: Training labels
            - y_test: Test labels
            - X_train_raw: Original training features (unscaled)
            - X_test_raw: Original test features (unscaled)
            - feature_names: Names of the 4 features
            - target_names: Names of the 3 classes
            - description: Dataset description

    Raises:
        ValueError: If test_size is invalid
    """
    # Load Iris dataset from DatasetManager
    dataset = DatasetManager.get_iris(test_size=test_size, random_state=random_state)

    # Extract components
    X_train_raw = dataset['X_train']
    X_test_raw = dataset['X_test']
    y_train = dataset['y_train']
    y_test = dataset['y_test']

    # Scale features for better KNN performance
    # KNN is sensitive to feature scales because it uses distance metrics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'feature_names': dataset['feature_names'],
        'target_names': dataset['target_names'],
        'description': dataset['description']
    }


def get_iris_dataset(test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
    """Convenience function to get the Iris dataset.

    This is a simpler interface to load_and_prepare_data for quick access.

    Args:
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with dataset components and metadata
    """
    return load_and_prepare_data(test_size=test_size, random_state=random_state)


def reduce_to_2d(X: np.ndarray, method: str = 'pca') -> np.ndarray:
    """Reduce high-dimensional data to 2D for visualization.

    Uses PCA (Principal Component Analysis) to project the 4-dimensional
    Iris features into 2D space while preserving maximum variance.

    Args:
        X: Input features array of shape (n_samples, n_features)
        method: Dimensionality reduction method ('pca' only for now)

    Returns:
        2D array of shape (n_samples, 2) with reduced features

    Raises:
        ValueError: If method is not supported or X has invalid shape
    """
    from sklearn.decomposition import PCA

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")

    if X.shape[1] < 2:
        raise ValueError(f"Cannot reduce data with {X.shape[1]} features to 2D")

    if method != 'pca':
        raise ValueError(f"Unsupported reduction method: {method}. Use 'pca'.")

    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    return X_2d
