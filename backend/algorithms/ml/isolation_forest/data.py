"""Data loading utilities for Isolation Forest algorithm."""

from typing import Dict, Any
import numpy as np
from utils.datasets import DatasetManager


def get_isolation_forest_data(
    n_samples: int = 300,
    n_outliers_ratio: float = 0.1,
    random_state: int = 42
) -> Dict[str, Any]:
    """Get data for Isolation Forest anomaly detection demonstration.

    Generates synthetic blob data with injected outliers to demonstrate
    Isolation Forest's ability to detect anomalies.

    The dataset consists of normal points clustered in blobs plus randomly
    distributed outlier points, which is ideal for showing Isolation Forest's
    strength in identifying anomalies without requiring labeled data.

    Args:
        n_samples: Total number of normal data points to generate
        n_outliers_ratio: Ratio of outliers to inject (relative to n_samples)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - X: Feature array (numpy array, shape [total_samples, 2])
            - y_true: Ground truth labels (1 for normal, -1 for outlier)
            - n_outliers: Actual number of outliers injected

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> data = get_isolation_forest_data(n_samples=500, n_outliers_ratio=0.15)
        >>> X = data['X']
        >>> print(f"Generated {len(X)} total samples")
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if not 0.0 <= n_outliers_ratio <= 0.5:
        raise ValueError(f"n_outliers_ratio must be between 0.0 and 0.5, got {n_outliers_ratio}")

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Calculate number of outliers to inject
    n_outliers = int(n_samples * n_outliers_ratio)

    # Use DatasetManager to generate clustered blob data (normal points)
    dataset = DatasetManager.get_blobs(
        n_samples=n_samples,
        centers=3,
        cluster_std=0.5,
        random_state=random_state
    )
    X_normal = dataset['X']

    # Generate outliers: random points spread across a wider range
    # Find the bounds of normal data
    x_min, x_max = X_normal[:, 0].min() - 2, X_normal[:, 0].max() + 2
    y_min, y_max = X_normal[:, 1].min() - 2, X_normal[:, 1].max() + 2

    # Generate random outliers outside the normal data distribution
    X_outliers = np.random.uniform(
        low=[x_min, y_min],
        high=[x_max, y_max],
        size=(n_outliers, 2)
    )

    # Combine normal and outlier data
    X = np.vstack([X_normal, X_outliers])

    # Create ground truth labels: 1 for normal, -1 for outliers
    y_true = np.ones(len(X), dtype=int)
    y_true[-n_outliers:] = -1

    # Shuffle the data to mix normal and outlier points
    shuffle_indices = np.random.permutation(len(X))
    X = X[shuffle_indices]
    y_true = y_true[shuffle_indices]

    return {
        'X': X,
        'y_true': y_true,
        'n_outliers': n_outliers
    }


def prepare_visualization_data(
    X: np.ndarray,
    predictions: np.ndarray,
    anomaly_scores: np.ndarray,
    y_true: np.ndarray = None
) -> Dict[str, Any]:
    """Prepare data for scatter plot visualization.

    Formats the anomaly detection results for frontend visualization,
    organizing points by their classification (normal vs anomaly).
    Isolation Forest predicts -1 for anomalies and 1 for normal points.

    Args:
        X: Feature array (shape [n_samples, 2])
        predictions: Anomaly predictions (-1 for anomaly, 1 for normal)
        anomaly_scores: Anomaly scores (more negative = more anomalous)
        y_true: Optional ground truth labels for comparison

    Returns:
        Dictionary containing:
            - normal_points: List of normal point data
            - anomaly_points: List of anomaly point data
            - anomaly_scores: List of anomaly scores for all points
            - score_stats: Statistics about anomaly scores

    Example:
        >>> viz_data = prepare_visualization_data(X, predictions, scores)
        >>> print(f"Detected {len(viz_data['anomaly_points'])} anomalies")
    """
    normal_points = []
    anomaly_points = []
    score_list = []

    # Separate points by prediction
    for i, (point, pred, score) in enumerate(zip(X, predictions, anomaly_scores)):
        point_data = {
            'x': float(point[0]),
            'y': float(point[1]),
            'score': float(score),
            'prediction': int(pred)
        }

        # Add ground truth if available
        if y_true is not None:
            point_data['true_label'] = int(y_true[i])

        if pred == -1:  # Anomaly
            anomaly_points.append(point_data)
        else:  # Normal
            normal_points.append(point_data)

        score_list.append({
            'index': i,
            'score': float(score),
            'prediction': int(pred)
        })

    # Calculate score statistics
    scores_array = anomaly_scores
    score_stats = {
        'min': float(scores_array.min()),
        'max': float(scores_array.max()),
        'mean': float(scores_array.mean()),
        'median': float(np.median(scores_array)),
        'std': float(scores_array.std())
    }

    return {
        'normal_points': normal_points,
        'anomaly_points': anomaly_points,
        'anomaly_scores': score_list,
        'score_stats': score_stats,
        'n_normal': len(normal_points),
        'n_anomalies': len(anomaly_points)
    }
