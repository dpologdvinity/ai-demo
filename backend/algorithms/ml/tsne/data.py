"""Data loading and preparation for t-SNE."""

from typing import Dict, Any
from backend.utils.datasets import DatasetManager


def load_digits_data() -> Dict[str, Any]:
    """Load and prepare the digits dataset for t-SNE visualization.

    The digits dataset contains 1,797 samples of 8x8 grayscale images of handwritten
    digits (0-9). Each image is represented as a 64-element feature vector. t-SNE can
    effectively reduce these 64 dimensions to 2D or 3D for visualization while
    preserving the local neighborhood structure, allowing similar digits to cluster
    together in the low-dimensional space.

    Returns:
        Dictionary containing:
            - X_train: Training features (numpy array, shape [n_samples, 64])
            - X_test: Test features (numpy array, shape [n_samples, 64])
            - y_train: Training labels (numpy array)
            - y_test: Test labels (numpy array)
            - images: Original 8x8 images (numpy array)
            - description: Dataset description string
    """
    return DatasetManager.get_digits(test_size=0.3, random_state=42)


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the digits dataset.

    Returns:
        Dictionary containing dataset metadata including number of samples,
        original features, and target classes.
    """
    data = load_digits_data()
    return {
        'name': 'Digits Dataset',
        'n_samples': len(data['X_train']) + len(data['X_test']),
        'n_features': data['X_train'].shape[1],
        'n_classes': len(set(data['y_train'])),
        'description': (
            'Handwritten digits dataset (8x8 images) with 64 features per sample. '
            'Perfect for demonstrating t-SNE\'s ability to reveal cluster structure '
            'in high-dimensional data.'
        )
    }
