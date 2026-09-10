"""Data loading utilities for Linear Regression.

This module provides functions to load and prepare datasets specifically
for Linear Regression demonstrations, primarily using the California Housing
dataset as a replacement for the deprecated Boston Housing dataset.
"""

from typing import Dict, Any
from utils.datasets import DatasetManager


def load_linear_regression_data(
    dataset_name: str = "boston",
    test_size: float = 0.3,
    normalize: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """Load and prepare data for Linear Regression.

    This function loads a regression dataset (default: California Housing)
    and optionally normalizes the features for better model performance.

    Args:
        dataset_name: Name of the dataset to load. Currently supports:
            - 'boston' or 'california': California Housing dataset (default)
        test_size: Proportion of the dataset to include in the test split.
            Must be between 0.1 and 0.5. Default is 0.3.
        normalize: Whether to normalize features using StandardScaler.
            Recommended for Linear Regression when features have different scales.
        random_state: Random seed for reproducible splits. Default is 42.

    Returns:
        Dictionary containing:
            - X_train: Training features (numpy array)
            - X_test: Test features (numpy array)
            - y_train: Training target values (numpy array)
            - y_test: Test target values (numpy array)
            - feature_names: List of feature names
            - description: Dataset description string
            - normalized: Boolean indicating if normalization was applied

    Raises:
        ValueError: If dataset_name is not supported or test_size is invalid.

    Example:
        >>> data = load_linear_regression_data(
        ...     dataset_name="boston",
        ...     test_size=0.3,
        ...     normalize=True
        ... )
        >>> X_train, y_train = data['X_train'], data['y_train']
    """
    if test_size < 0.1 or test_size > 0.5:
        raise ValueError(
            f"test_size must be between 0.1 and 0.5, got {test_size}"
        )

    # Load dataset
    if dataset_name in ["boston", "california"]:
        dataset = DatasetManager.get_boston(
            test_size=test_size,
            random_state=random_state
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: boston, california"
        )

    # Normalize features if requested
    if normalize:
        X_train_normalized, X_test_normalized = DatasetManager.normalize_data(
            dataset['X_train'],
            dataset['X_test']
        )
        dataset['X_train'] = X_train_normalized
        dataset['X_test'] = X_test_normalized
        dataset['normalized'] = True
    else:
        dataset['normalized'] = False

    return dataset


def get_available_datasets() -> list[str]:
    """Get list of available datasets for Linear Regression.

    Returns:
        List of dataset names that can be used with load_linear_regression_data().
    """
    return ["boston", "california"]
