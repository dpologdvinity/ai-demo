"""Data loading utilities for K-Means Clustering algorithm."""

from typing import Dict, Any
import numpy as np
from utils.datasets import DatasetManager


def get_kmeans_data(
    n_samples: int = 300,
    n_clusters: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """Get data for K-Means clustering demonstration.

    Uses the DatasetManager to generate synthetic blob clusters suitable
    for demonstrating K-Means clustering algorithm.

    Args:
        n_samples: Total number of data points to generate
        n_clusters: Number of ground-truth clusters to generate
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - X: Feature array (numpy array, shape [n_samples, 2])
            - y: Ground truth cluster labels (numpy array)
            - n_samples: Number of samples generated
            - n_centers: Number of cluster centers

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> data = get_kmeans_data(n_samples=500, n_clusters=4)
        >>> X, y = data['X'], data['y']
        >>> print(f"Generated {len(X)} samples with {data['n_centers']} clusters")
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be at least 2, got {n_clusters}")

    # Use DatasetManager to generate blob clusters
    dataset = DatasetManager.get_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        random_state=random_state
    )

    return dataset


def prepare_visualization_data(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray
) -> Dict[str, Any]:
    """Prepare data for scatter plot visualization.

    Formats the clustering results for frontend visualization, organizing
    points by cluster assignment and including centroid information.

    Args:
        X: Feature array (shape [n_samples, 2])
        labels: Cluster assignments for each point (shape [n_samples])
        centers: Cluster centroids (shape [n_clusters, 2])

    Returns:
        Dictionary containing:
            - clusters: List of cluster data arrays, each with x, y, cluster fields
            - centroids: List of centroid coordinates with x, y, cluster fields
            - n_clusters: Number of clusters

    Example:
        >>> viz_data = prepare_visualization_data(X, labels, centers)
        >>> print(f"Visualizing {viz_data['n_clusters']} clusters")
    """
    n_clusters = len(centers)
    clusters = []

    # Group points by cluster
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_points = X[cluster_mask]

        cluster_data = [
            {
                'x': float(point[0]),
                'y': float(point[1]),
                'cluster': int(i)
            }
            for point in cluster_points
        ]
        clusters.append(cluster_data)

    # Format centroids
    centroids = [
        {
            'x': float(center[0]),
            'y': float(center[1]),
            'cluster': int(i)
        }
        for i, center in enumerate(centers)
    ]

    return {
        'clusters': clusters,
        'centroids': centroids,
        'n_clusters': n_clusters
    }
