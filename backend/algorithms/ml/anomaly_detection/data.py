"""Data loading utilities for Anomaly Detection algorithm."""

from typing import Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.datasets import DatasetManager


def get_anomaly_detection_data(
    dataset_type: str = 'synthetic',
    n_samples: int = 300,
    contamination: float = 0.1,
    random_state: int = 42
) -> Dict[str, Any]:
    """Get data for anomaly detection demonstration.

    Generates or loads datasets with injected anomalies to demonstrate
    anomaly detection algorithms.

    Args:
        dataset_type: Type of dataset ('synthetic', 'fraud', 'network')
        n_samples: Number of normal samples to generate
        contamination: Ratio of anomalies to inject
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - X: Feature array (numpy array)
            - y_true: Ground truth labels (1 for normal, -1 for anomaly)
            - feature_names: List of feature names
            - description: Dataset description

    Raises:
        ValueError: If parameters are invalid
    """
    np.random.seed(random_state)

    if dataset_type == 'synthetic':
        return _generate_synthetic_data(n_samples, contamination, random_state)
    elif dataset_type == 'fraud':
        return _generate_fraud_data(n_samples, contamination, random_state)
    elif dataset_type == 'network':
        return _generate_network_data(n_samples, contamination, random_state)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def _generate_synthetic_data(
    n_samples: int,
    contamination: float,
    random_state: int
) -> Dict[str, Any]:
    """Generate synthetic 2D data with outliers.

    Creates clustered normal data with randomly distributed outliers.
    """
    np.random.seed(random_state)
    n_outliers = int(n_samples * contamination)

    # Generate normal data as clustered blobs
    dataset = DatasetManager.get_blobs(
        n_samples=n_samples,
        centers=3,
        random_state=random_state
    )
    X_normal = dataset['X']

    # Generate outliers spread across wider range
    x_min, x_max = X_normal[:, 0].min() - 3, X_normal[:, 0].max() + 3
    y_min, y_max = X_normal[:, 1].min() - 3, X_normal[:, 1].max() + 3

    X_outliers = np.random.uniform(
        low=[x_min, y_min],
        high=[x_max, y_max],
        size=(n_outliers, 2)
    )

    # Combine and shuffle
    X = np.vstack([X_normal, X_outliers])
    y_true = np.ones(len(X), dtype=int)
    y_true[-n_outliers:] = -1

    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y_true = y_true[shuffle_idx]

    return {
        'X': X,
        'y_true': y_true,
        'feature_names': ['Feature 1', 'Feature 2'],
        'description': 'Synthetic 2D data with clustered normal points and random outliers'
    }


def _generate_fraud_data(
    n_samples: int,
    contamination: float,
    random_state: int
) -> Dict[str, Any]:
    """Generate fraud detection dataset.

    Simulates transaction features: amount, time, location distance, etc.
    """
    np.random.seed(random_state)
    n_outliers = int(n_samples * contamination)

    # Normal transactions: typical amounts, times, patterns
    # Feature 1: Transaction amount (log scale)
    normal_amounts = np.random.lognormal(mean=3.5, sigma=1.0, size=n_samples)

    # Feature 2: Time of day (0-24 hours, normal distribution around business hours)
    normal_times = np.random.normal(loc=14, scale=4, size=n_samples) % 24

    # Feature 3: Location distance (km from home)
    normal_distances = np.abs(np.random.normal(loc=5, scale=10, size=n_samples))

    # Feature 4: Transaction frequency (transactions per day)
    normal_frequency = np.random.gamma(shape=2, scale=2, size=n_samples)

    X_normal = np.column_stack([
        normal_amounts,
        normal_times,
        normal_distances,
        normal_frequency
    ])

    # Fraudulent transactions: unusual patterns
    fraud_amounts = np.random.uniform(100, 5000, size=n_outliers)  # Very high amounts
    fraud_times = np.random.uniform(0, 6, size=n_outliers)  # Late night/early morning
    fraud_distances = np.random.uniform(500, 2000, size=n_outliers)  # Far from home
    fraud_frequency = np.random.uniform(10, 50, size=n_outliers)  # High frequency

    X_fraud = np.column_stack([
        fraud_amounts,
        fraud_times,
        fraud_distances,
        fraud_frequency
    ])

    # Combine and shuffle
    X = np.vstack([X_normal, X_fraud])
    y_true = np.ones(len(X), dtype=int)
    y_true[-n_outliers:] = -1

    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y_true = y_true[shuffle_idx]

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # For visualization, use first 2 principal features
    X_2d = X[:, :2]

    return {
        'X': X_2d,
        'y_true': y_true,
        'feature_names': ['Transaction Amount (normalized)', 'Transaction Time (normalized)'],
        'description': 'Credit card fraud detection: normal vs fraudulent transactions'
    }


def _generate_network_data(
    n_samples: int,
    contamination: float,
    random_state: int
) -> Dict[str, Any]:
    """Generate network intrusion detection dataset.

    Simulates network connection features: packet size, duration, etc.
    """
    np.random.seed(random_state)
    n_outliers = int(n_samples * contamination)

    # Normal connections
    # Feature 1: Connection duration (seconds)
    normal_duration = np.abs(np.random.normal(loc=10, scale=5, size=n_samples))

    # Feature 2: Data transfer rate (KB/s)
    normal_rate = np.abs(np.random.normal(loc=50, scale=20, size=n_samples))

    # Feature 3: Number of packets
    normal_packets = np.random.poisson(lam=100, size=n_samples)

    # Feature 4: Error rate (%)
    normal_errors = np.abs(np.random.normal(loc=0.5, scale=0.3, size=n_samples))

    X_normal = np.column_stack([
        normal_duration,
        normal_rate,
        normal_packets,
        normal_errors
    ])

    # Anomalous connections (attacks, intrusions)
    attack_duration = np.random.uniform(0.1, 1, size=n_outliers)  # Very short
    attack_rate = np.random.uniform(500, 2000, size=n_outliers)  # Very high rate
    attack_packets = np.random.uniform(1000, 5000, size=n_outliers)  # Flood
    attack_errors = np.random.uniform(5, 20, size=n_outliers)  # High error rate

    X_attack = np.column_stack([
        attack_duration,
        attack_rate,
        attack_packets,
        attack_errors
    ])

    # Combine and shuffle
    X = np.vstack([X_normal, X_attack])
    y_true = np.ones(len(X), dtype=int)
    y_true[-n_outliers:] = -1

    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y_true = y_true[shuffle_idx]

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # For visualization, use first 2 features
    X_2d = X[:, :2]

    return {
        'X': X_2d,
        'y_true': y_true,
        'feature_names': ['Connection Duration (normalized)', 'Data Transfer Rate (normalized)'],
        'description': 'Network intrusion detection: normal vs malicious connections'
    }


def prepare_visualization_data(
    X: np.ndarray,
    predictions: np.ndarray,
    anomaly_scores: np.ndarray,
    y_true: np.ndarray = None,
    method_name: str = None
) -> Dict[str, Any]:
    """Prepare data for visualization.

    Args:
        X: Feature array (shape [n_samples, 2])
        predictions: Anomaly predictions (-1 for anomaly, 1 for normal)
        anomaly_scores: Anomaly scores (more negative = more anomalous)
        y_true: Optional ground truth labels
        method_name: Name of the detection method

    Returns:
        Dictionary containing visualization data
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
            'prediction': int(pred),
            'index': i
        }

        if y_true is not None:
            point_data['true_label'] = int(y_true[i])
            point_data['correct'] = int(pred) == int(y_true[i])

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
    score_stats = {
        'min': float(anomaly_scores.min()),
        'max': float(anomaly_scores.max()),
        'mean': float(anomaly_scores.mean()),
        'median': float(np.median(anomaly_scores)),
        'std': float(anomaly_scores.std()),
        'threshold': float(np.percentile(anomaly_scores, 10))  # 10th percentile as threshold
    }

    # Score distribution histogram
    hist, bin_edges = np.histogram(anomaly_scores, bins=30)
    score_distribution = [
        {'bin_center': float((bin_edges[i] + bin_edges[i+1]) / 2), 'count': int(hist[i])}
        for i in range(len(hist))
    ]

    result = {
        'normal_points': normal_points,
        'anomaly_points': anomaly_points,
        'anomaly_scores': score_list,
        'score_stats': score_stats,
        'score_distribution': score_distribution,
        'n_normal': len(normal_points),
        'n_anomalies': len(anomaly_points)
    }

    if method_name:
        result['method_name'] = method_name

    return result


def get_dataset_info(dataset_type: str = 'synthetic') -> Dict[str, Any]:
    """Get information about a dataset.

    Args:
        dataset_type: Type of dataset

    Returns:
        Dictionary with dataset information
    """
    dataset_descriptions = {
        'synthetic': {
            'name': 'Synthetic 2D Data',
            'description': 'Clustered normal points with random outliers',
            'features': ['Feature 1', 'Feature 2'],
            'n_features': 2,
            'recommended_contamination': 0.1
        },
        'fraud': {
            'name': 'Credit Card Fraud',
            'description': 'Transaction features: amount, time, location, frequency',
            'features': ['Transaction Amount', 'Transaction Time', 'Distance', 'Frequency'],
            'n_features': 4,
            'recommended_contamination': 0.1
        },
        'network': {
            'name': 'Network Intrusion',
            'description': 'Connection features: duration, transfer rate, packets, errors',
            'features': ['Duration', 'Transfer Rate', 'Packets', 'Error Rate'],
            'n_features': 4,
            'recommended_contamination': 0.15
        }
    }

    return dataset_descriptions.get(dataset_type, dataset_descriptions['synthetic'])
