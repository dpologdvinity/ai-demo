"""Data loading utilities for DBSCAN Clustering algorithm."""

from typing import Dict, Any
import numpy as np
from backend.utils.datasets import DatasetManager


def get_dbscan_data(
    n_samples: int = 300,
    noise: float = 0.1,
    random_state: int = 42
) -> Dict[str, Any]:
    """Get data for DBSCAN clustering demonstration.

    Uses the DatasetManager to generate synthetic moons dataset suitable
    for demonstrating DBSCAN's ability to find non-convex clusters.

    The moons dataset consists of two interleaving half circles, which is
    ideal for showing DBSCAN's strength in discovering arbitrarily shaped
    clusters that traditional algorithms like K-Means struggle with.

    Args:
        n_samples: Total number of data points to generate
        noise: Standard deviation of Gaussian noise added to the data
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - X: Feature array (numpy array, shape [n_samples, 2])
            - y: Ground truth labels (numpy array)

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> data = get_dbscan_data(n_samples=500, noise=0.15)
        >>> X = data['X']
        >>> print(f"Generated {len(X)} samples")
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if noise < 0:
        raise ValueError(f"noise must be non-negative, got {noise}")

    # Use DatasetManager to generate moons dataset
    dataset = DatasetManager.get_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )

    return dataset


def prepare_visualization_data(
    X: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """Prepare data for scatter plot visualization.

    Formats the clustering results for frontend visualization, organizing
    points by cluster assignment. DBSCAN assigns -1 to noise points, which
    are visualized separately.

    Args:
        X: Feature array (shape [n_samples, 2])
        labels: Cluster assignments for each point (shape [n_samples])
            Noise points are labeled as -1

    Returns:
        Dictionary containing:
            - clusters: List of cluster data arrays, each with x, y, cluster fields
            - noise_points: Array of noise point coordinates
            - n_clusters: Number of clusters found (excluding noise)

    Example:
        >>> viz_data = prepare_visualization_data(X, labels)
        >>> print(f"Found {viz_data['n_clusters']} clusters")
        >>> print(f"Noise points: {len(viz_data['noise_points'])}")
    """
    # Get unique cluster labels (excluding noise which is labeled -1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    clusters = []
    noise_points = []

    # Group points by cluster
    for label in unique_labels:
        if label == -1:
            # Handle noise points separately
            noise_mask = labels == -1
            noise_pts = X[noise_mask]
            noise_points = [
                {
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'cluster': -1
                }
                for point in noise_pts
            ]
        else:
            # Regular cluster points
            cluster_mask = labels == label
            cluster_points = X[cluster_mask]

            cluster_data = [
                {
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'cluster': int(label)
                }
                for point in cluster_points
            ]
            clusters.append(cluster_data)

    return {
        'clusters': clusters,
        'noise_points': noise_points,
        'n_clusters': n_clusters
    }
