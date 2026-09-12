"""Data utilities for Learning Rate Scheduling demonstration."""

from typing import Dict, Any


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the synthetic training task dataset.

    Returns:
        Dictionary containing dataset information for API responses
    """
    return {
        "name": "Synthetic Classification Task",
        "description": (
            "A synthetic binary classification task with 1000 samples. "
            "The dataset is designed to demonstrate how different learning rate "
            "schedules affect convergence speed and final performance. "
            "Features are 20-dimensional with non-linear decision boundaries."
        ),
        "source": "Generated using sklearn.datasets.make_classification",
        "n_samples": 1000,
        "n_features": 20,
        "n_classes": 2,
        "train_size": 800,
        "test_size": 200,
        "task_type": "Binary Classification",
        "features": [
            "20 informative features",
            "Non-linear decision boundary",
            "Some redundant features",
            "Moderate class separation"
        ],
        "preprocessing": [
            "Standardization (zero mean, unit variance)",
            "Train-test split (80/20)",
            "Fixed random seed for reproducibility"
        ]
    }
