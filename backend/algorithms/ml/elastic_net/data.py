"""Data loading and preparation for Elastic Net Regression."""

from typing import Dict, Any
from backend.utils.datasets import DatasetManager


def load_housing_data() -> Dict[str, Any]:
    """Load and prepare the California Housing dataset for Elastic Net Regression.

    The California Housing dataset (replacement for Boston Housing) contains
    20,640 samples with 8 features predicting median house values. This dataset
    is well-suited for demonstrating Elastic Net's combined regularization as
    it contains features with varying importance and some correlation.

    Returns:
        Dictionary containing:
            - X_train: Training features (numpy array)
            - X_test: Test features (numpy array)
            - y_train: Training target values (numpy array)
            - y_test: Test target values (numpy array)
            - feature_names: List of feature names
            - description: Dataset description string
    """
    return DatasetManager.get_boston(test_size=0.3, random_state=42)


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the California Housing dataset.

    Returns:
        Dictionary containing dataset metadata including number of samples,
        features, and feature names.
    """
    data = load_housing_data()
    return {
        'name': 'California Housing Dataset',
        'n_samples': len(data['X_train']) + len(data['X_test']),
        'n_features': len(data['feature_names']),
        'feature_names': data['feature_names'],
        'target_description': 'Median house value (in $100,000s)',
        'description': 'California housing prices with geographic and demographic features'
    }
