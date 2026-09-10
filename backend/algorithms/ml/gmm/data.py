"""Data loading and preparation for Gaussian Mixture Model.

This module provides functions to generate synthetic blob datasets suitable
for demonstrating GMM clustering capabilities.
"""

from typing import Dict, Any
import numpy as np
from utils.datasets import DatasetManager


def load_blobs_data(
    n_samples: int = 300,
    centers: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """Load and prepare blob dataset for GMM clustering.

    Generates isotropic Gaussian blobs that are ideal for demonstrating
    Gaussian Mixture Model clustering. The dataset contains 2D points
    distributed across multiple Gaussian clusters.

    Args:
        n_samples: Total number of samples to generate. Default: 300
        centers: Number of cluster centers to generate. Default: 3
        random_state: Random seed for reproducibility. Default: 42

    Returns:
        Dictionary containing:
            - X: Feature array (numpy array, shape [n_samples, 2])
            - y: True cluster labels (numpy array)
            - n_samples: Number of samples generated
            - n_centers: Number of cluster centers
            - description: Dataset description string

    Raises:
        ValueError: If n_samples or centers is less than 1.

    Example:
        >>> data = load_blobs_data(n_samples=300, centers=3)
        >>> X, y = data['X'], data['y']
        >>> print(f"Generated {len(X)} samples with {data['n_centers']} clusters")
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if centers < 1:
        raise ValueError(f"centers must be at least 1, got {centers}")

    blobs_data = DatasetManager.get_blobs(
        n_samples=n_samples,
        centers=centers,
        random_state=random_state
    )

    return {
        'X': blobs_data['X'],
        'y': blobs_data['y'],
        'n_samples': blobs_data['n_samples'],
        'n_centers': blobs_data['n_centers'],
        'description': (
            f"Synthetic dataset with {n_samples} samples distributed across "
            f"{centers} Gaussian clusters. Each cluster represents an isotropic "
            f"Gaussian blob in 2D space, ideal for demonstrating GMM's ability "
            f"to identify and model mixture components."
        )
    }


def get_dataset_info(
    n_samples: int = 300,
    centers: int = 3
) -> Dict[str, Any]:
    """Get information about the blob dataset.

    Provides metadata about the generated blob dataset without actually
    generating the data.

    Args:
        n_samples: Number of samples that would be generated. Default: 300
        centers: Number of cluster centers that would be used. Default: 3

    Returns:
        Dictionary containing dataset metadata including type, dimensions,
        number of samples, and number of clusters.

    Example:
        >>> info = get_dataset_info(n_samples=300, centers=3)
        >>> print(f"Dataset: {info['name']}")
        >>> print(f"Samples: {info['n_samples']}, Features: {info['n_features']}")
    """
    return {
        'name': 'Gaussian Blobs Dataset',
        'type': 'synthetic',
        'n_samples': n_samples,
        'n_features': 2,
        'n_clusters': centers,
        'description': (
            'Synthetic dataset of isotropic Gaussian blobs suitable for '
            'clustering algorithms. Each blob represents a distinct Gaussian '
            'distribution in 2D space.'
        ),
        'use_case': (
            'Ideal for demonstrating Gaussian Mixture Models and other '
            'clustering algorithms that assume Gaussian-distributed data.'
        )
    }


def prepare_data_for_gmm(X: np.ndarray) -> np.ndarray:
    """Prepare data for GMM training.

    GMM generally works well without extensive preprocessing, but this
    function provides a place for any necessary data preparation steps.

    Args:
        X: Raw feature data, shape (n_samples, n_features)

    Returns:
        Prepared feature data, ready for GMM training.

    Note:
        Currently returns data unchanged as GMM is scale-invariant when
        using 'full' covariance type. For other covariance types, consider
        standardization if features have very different scales.

    Example:
        >>> X = load_blobs_data()['X']
        >>> X_prepared = prepare_data_for_gmm(X)
    """
    # GMM doesn't require standardization for 'full' covariance
    # But we validate the data shape
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")

    if X.size == 0:
        raise ValueError("X cannot be empty")

    return X


def validate_data_for_visualization(X: np.ndarray) -> None:
    """Validate that data is suitable for 2D visualization.

    Args:
        X: Feature data to validate

    Raises:
        ValueError: If data doesn't have exactly 2 features.

    Example:
        >>> X = load_blobs_data()['X']
        >>> validate_data_for_visualization(X)  # Passes silently
    """
    if X.shape[1] != 2:
        raise ValueError(
            f"Visualization requires 2D data, got {X.shape[1]} features. "
            f"Consider using dimensionality reduction (PCA, t-SNE) for higher dimensions."
        )
