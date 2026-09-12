"""Data loading and preparation for Naive Bayes algorithm.

This module provides convenient access to datasets suitable for
Naive Bayes classification demonstrations.
"""

from typing import Dict, Any
from backend.utils.datasets import DatasetManager


def load_iris_data(test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
    """Load Iris dataset for Naive Bayes classification.

    The Iris dataset is ideal for Naive Bayes demonstrations due to its
    clean separation between classes and continuous features.

    Args:
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducible splits

    Returns:
        Dictionary containing train/test splits and metadata:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
            - feature_names: Names of features
            - target_names: Names of target classes
            - description: Dataset description

    Example:
        >>> data = load_iris_data(test_size=0.2)
        >>> X_train, y_train = data['X_train'], data['y_train']
    """
    return DatasetManager.get_iris(test_size=test_size, random_state=random_state)


def load_wine_data(test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
    """Load Wine dataset for Naive Bayes classification.

    The Wine dataset contains chemical analysis of wines from three different
    cultivars, suitable for multi-class classification.

    Args:
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducible splits

    Returns:
        Dictionary containing train/test splits and metadata

    Example:
        >>> data = load_wine_data(test_size=0.25)
        >>> X_train, y_train = data['X_train'], data['y_train']
    """
    return DatasetManager.get_wine(test_size=test_size, random_state=random_state)


def load_digits_data(test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
    """Load Digits dataset for Naive Bayes classification.

    The handwritten digits dataset (8x8 images flattened to 64 features)
    provides a more challenging classification task with 10 classes.

    Args:
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducible splits

    Returns:
        Dictionary containing train/test splits and metadata

    Example:
        >>> data = load_digits_data(test_size=0.2)
        >>> X_train, y_train = data['X_train'], data['y_train']
    """
    return DatasetManager.get_digits(test_size=test_size, random_state=random_state)


def get_supported_datasets() -> list:
    """Get list of datasets supported by Naive Bayes algorithm.

    Returns:
        List of supported dataset names

    Example:
        >>> datasets = get_supported_datasets()
        >>> print(datasets)
        ['iris', 'wine', 'digits']
    """
    return ['iris', 'wine', 'digits']
