"""Data loading and preparation for Hierarchical Clustering demonstrations.

This module provides functions to load and prepare the blobs dataset for
hierarchical clustering demonstrations.
"""

from typing import Dict, Any
import numpy as np
from backend.utils.datasets import DatasetManager


def load_blobs_data(
    n_samples: int = 300,
    n_centers: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """Load blob clusters dataset for hierarchical clustering.

    Generates isotropic Gaussian blobs for clustering demonstrations.
    This is ideal for hierarchical clustering as it creates well-separated
    clusters that form a natural hierarchy.

    Args:
        n_samples: Total number of points to generate (default: 300)
        n_centers: Number of cluster centers (default: 3)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary containing:
            - X: Feature array (numpy array, shape [n_samples, 2])
            - y: True cluster labels (numpy array)
            - n_samples: Number of samples generated
            - n_centers: Number of cluster centers

    Raises:
        ValueError: If n_samples or n_centers is less than 1
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if n_centers < 1:
        raise ValueError(f"n_centers must be at least 1, got {n_centers}")

    dataset = DatasetManager.get_blobs(
        n_samples=n_samples,
        centers=n_centers,
        random_state=random_state
    )

    return dataset


def prepare_clustering_data(X: np.ndarray) -> np.ndarray:
    """Prepare data for hierarchical clustering.

    Performs any necessary preprocessing for hierarchical clustering.
    Currently returns data as-is since the blobs dataset is already
    well-scaled, but this function can be extended for other preprocessing.

    Args:
        X: Raw feature data

    Returns:
        Preprocessed feature data

    Raises:
        ValueError: If X is empty
    """
    if X.size == 0:
        raise ValueError("X cannot be empty")

    # For blob data, no additional preprocessing is needed
    # The data is already well-scaled
    return X


def get_sample_data(n_samples: int = 300, n_clusters: int = 3) -> Dict[str, Any]:
    """Get sample data for hierarchical clustering demonstration.

    Convenience function that loads and prepares blob data with
    the specified number of samples and clusters.

    Args:
        n_samples: Number of samples to generate (default: 300)
        n_clusters: Number of clusters to generate (default: 3)

    Returns:
        Dictionary containing prepared dataset

    Example:
        >>> data = get_sample_data(n_samples=200, n_clusters=4)
        >>> X = data['X']
        >>> # Use X for training hierarchical clustering
    """
    dataset = load_blobs_data(
        n_samples=n_samples,
        n_centers=n_clusters,
        random_state=42
    )

    # Prepare the data
    dataset['X'] = prepare_clustering_data(dataset['X'])

    return dataset
