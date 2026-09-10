"""Data loading and preparation for Random Forest."""

from typing import Dict, Any
from utils.datasets import DatasetManager


def load_wine_data() -> Dict[str, Any]:
    """Load and prepare the wine dataset for Random Forest classification.

    The wine dataset contains 178 samples with 13 chemical features and 3 target
    classes representing different wine cultivars. This dataset is well-suited for
    demonstrating Random Forest's ability to handle multi-class classification with
    multiple features.

    Returns:
        Dictionary containing:
            - X_train: Training features (numpy array)
            - X_test: Test features (numpy array)
            - y_train: Training labels (numpy array)
            - y_test: Test labels (numpy array)
            - feature_names: List of feature names
            - target_names: List of target class names
            - description: Dataset description string
    """
    return DatasetManager.get_wine(test_size=0.3, random_state=42)


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the wine dataset.

    Returns:
        Dictionary containing dataset metadata including number of samples,
        features, classes, and feature names.
    """
    data = load_wine_data()
    return {
        'name': 'Wine Dataset',
        'n_samples': len(data['X_train']) + len(data['X_test']),
        'n_features': len(data['feature_names']),
        'n_classes': len(data['target_names']),
        'feature_names': data['feature_names'],
        'target_names': data['target_names'],
        'description': 'Wine recognition dataset with chemical analysis of wines grown in the same region in Italy'
    }
