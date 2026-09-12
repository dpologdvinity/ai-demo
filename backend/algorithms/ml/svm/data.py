"""Data loading and preparation for Support Vector Machine."""

from typing import Dict, Any
from backend.utils.datasets import DatasetManager


def load_iris_data() -> Dict[str, Any]:
    """Load and prepare the Iris dataset for SVM classification.

    The Iris dataset contains 150 samples with 4 features (sepal length, sepal width,
    petal length, petal width) and 3 target classes (setosa, versicolor, virginica).
    This dataset is ideal for demonstrating SVM's ability to find optimal hyperplanes
    for multi-class classification.

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
    return DatasetManager.get_iris(test_size=0.3, random_state=42)


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the Iris dataset.

    Returns:
        Dictionary containing dataset metadata including number of samples,
        features, classes, and feature names.
    """
    data = load_iris_data()
    return {
        'name': 'Iris Dataset',
        'n_samples': len(data['X_train']) + len(data['X_test']),
        'n_features': len(data['feature_names']),
        'n_classes': len(data['target_names']),
        'feature_names': data['feature_names'],
        'target_names': data['target_names'],
        'description': 'Classic iris flower dataset with measurements of sepal and petal dimensions'
    }
