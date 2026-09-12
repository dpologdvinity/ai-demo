"""Data preparation utilities for Imbalanced Classification.

This module handles creation of imbalanced datasets for demonstrating
various resampling techniques and class balancing strategies.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def prepare_imbalanced_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 15,
    imbalance_ratio: int = 10,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create an imbalanced binary classification dataset.

    Generates a synthetic dataset with significant class imbalance to
    demonstrate the effectiveness of various balancing techniques.

    Args:
        n_samples: Total number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        imbalance_ratio: Ratio of majority to minority class (e.g., 10 means 10:1)
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels (0=majority, 1=minority)
            - y_test: Test labels
            - metadata: Dataset metadata and statistics

    Example:
        >>> X_train, X_test, y_train, y_test, meta = prepare_imbalanced_data(
        ...     n_samples=1000, imbalance_ratio=10
        ... )
        >>> print(meta['class_distribution'])
        {0: 909, 1: 91}  # Approximately 10:1 ratio
    """
    # Calculate weights for imbalanced classes
    # For imbalance_ratio of 10, minority class should be ~9% (1/11)
    minority_weight = 1.0 / (imbalance_ratio + 1)
    weights = [1.0 - minority_weight, minority_weight]

    # Generate imbalanced classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_clusters_per_class=2,
        weights=weights,
        flip_y=0.01,  # Add small amount of noise
        random_state=random_state
    )

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution in splits
    )

    # Calculate class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    class_distribution_train = dict(zip(unique_train.tolist(), counts_train.tolist()))
    class_distribution_test = dict(zip(unique_test.tolist(), counts_test.tolist()))

    # Calculate actual imbalance ratio
    actual_ratio = (
        class_distribution_train.get(0, 0) / class_distribution_train.get(1, 1)
        if class_distribution_train.get(1, 0) > 0 else 0
    )

    # Prepare metadata
    metadata = {
        'n_samples_total': n_samples,
        'n_samples_train': X_train.shape[0],
        'n_samples_test': X_test.shape[0],
        'n_features': n_features,
        'n_informative': n_informative,
        'target_imbalance_ratio': imbalance_ratio,
        'actual_imbalance_ratio': float(actual_ratio),
        'class_distribution_train': class_distribution_train,
        'class_distribution_test': class_distribution_test,
        'minority_class': 1,
        'majority_class': 0,
        'feature_names': [f'feature_{i}' for i in range(n_features)]
    }

    return X_train, X_test, y_train, y_test, metadata


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the imbalanced classification dataset.

    Returns:
        Dictionary containing dataset description and characteristics
    """
    return {
        'name': 'Imbalanced Binary Classification',
        'description': (
            'Synthetic binary classification dataset with significant class imbalance. '
            'Designed to demonstrate the effectiveness of various resampling techniques '
            'such as SMOTE, undersampling, oversampling, and class weight balancing. '
            'The dataset simulates real-world scenarios like fraud detection, medical '
            'diagnosis, or anomaly detection where positive cases are rare.'
        ),
        'type': 'classification',
        'task': 'binary_classification',
        'n_classes': 2,
        'default_n_samples': 1000,
        'default_n_features': 20,
        'default_imbalance_ratio': 10,
        'features': {
            'type': 'continuous',
            'description': 'Synthetic features with varying degrees of informativeness'
        },
        'target': {
            'type': 'binary',
            'classes': ['Majority Class (0)', 'Minority Class (1)'],
            'description': 'Binary target with significant class imbalance'
        },
        'challenges': [
            'Severe class imbalance',
            'Model bias toward majority class',
            'Low recall for minority class',
            'Misleading accuracy metrics'
        ],
        'typical_use_cases': [
            'Fraud detection (1% fraud rate)',
            'Medical diagnosis (rare diseases)',
            'Anomaly detection in systems',
            'Quality control defect detection',
            'Churn prediction',
            'Credit default prediction'
        ]
    }


def calculate_class_distribution(y: np.ndarray) -> Dict[str, Any]:
    """Calculate class distribution statistics.

    Args:
        y: Target labels array

    Returns:
        Dictionary with class counts and ratio
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))

    class_0_count = distribution.get(0, 0)
    class_1_count = distribution.get(1, 0)
    ratio = class_0_count / class_1_count if class_1_count > 0 else 0

    return {
        'class_0': class_0_count,
        'class_1': class_1_count,
        'ratio': float(ratio),
        'total': len(y)
    }


def get_sampling_statistics(
    y_original: np.ndarray,
    y_resampled: np.ndarray,
    strategy: str
) -> Dict[str, Any]:
    """Get statistics about sampling transformation.

    Args:
        y_original: Original target labels
        y_resampled: Resampled target labels
        strategy: Name of the sampling strategy

    Returns:
        Dictionary with sampling statistics
    """
    original_dist = calculate_class_distribution(y_original)
    resampled_dist = calculate_class_distribution(y_resampled)

    return {
        'strategy': strategy,
        'original': original_dist,
        'resampled': resampled_dist,
        'samples_added': resampled_dist['total'] - original_dist['total'],
        'samples_removed': original_dist['total'] - resampled_dist['total'],
        'minority_increase': resampled_dist['class_1'] - original_dist['class_1'],
        'majority_decrease': original_dist['class_0'] - resampled_dist['class_0']
    }
