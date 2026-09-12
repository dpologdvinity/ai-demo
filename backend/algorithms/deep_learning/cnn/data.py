"""Data loading and preprocessing for CNN."""

from typing import Dict, Any
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_digits_data(
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True
) -> Dict[str, Any]:
    """Load and preprocess digits dataset for CNN.

    The digits dataset contains 8x8 grayscale images of handwritten digits (0-9).
    Images are reshaped to be suitable for CNN input.

    Args:
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        normalize: Whether to normalize pixel values

    Returns:
        Dictionary containing:
            - X_train: Training images (N, 8, 8)
            - X_test: Test images (M, 8, 8)
            - y_train: Training labels
            - y_test: Test labels
            - feature_names: List of feature descriptions
            - target_names: List of class names
            - num_classes: Number of classes
            - image_shape: Shape of individual images
    """
    # Load digits dataset
    digits = load_digits()
    X = digits.images  # Shape: (n_samples, 8, 8)
    y = digits.target

    # Normalize pixel values to [0, 1]
    if normalize:
        X = X / 16.0  # Digits dataset has values 0-16

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': [f'pixel_{i}_{j}' for i in range(8) for j in range(8)],
        'target_names': [str(i) for i in range(10)],
        'num_classes': 10,
        'image_shape': (8, 8)
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the digits dataset.

    Returns:
        Dictionary with dataset metadata
    """
    digits = load_digits()

    return {
        'name': 'Digits',
        'description': 'Handwritten digits (0-9) from UCI ML repository',
        'num_samples': len(digits.data),
        'num_features': 64,  # 8x8 images
        'num_classes': 10,
        'image_shape': '8x8 grayscale',
        'feature_range': '0-16 (normalized to 0-1)',
        'class_distribution': {
            str(i): int(np.sum(digits.target == i))
            for i in range(10)
        },
        'use_case': 'Digit recognition and image classification'
    }


def prepare_visualization_data(
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    feature_maps: np.ndarray
) -> Dict[str, Any]:
    """Prepare visualization data for frontend.

    Args:
        X_sample: Sample images
        y_sample: Sample labels
        feature_maps: Feature maps from CNN

    Returns:
        Dictionary with visualization data
    """
    # Convert images to list format
    sample_images = []
    for i in range(min(5, len(X_sample))):
        sample_images.append({
            'image': X_sample[i].tolist(),
            'label': int(y_sample[i])
        })

    # Convert feature maps to list format
    feature_maps_list = []
    if feature_maps is not None:
        # feature_maps shape: (num_samples, num_filters, H, W)
        for sample_idx in range(min(3, feature_maps.shape[0])):
            filters = []
            for filter_idx in range(min(8, feature_maps.shape[1])):
                filters.append({
                    'filter_idx': filter_idx,
                    'feature_map': feature_maps[sample_idx, filter_idx].tolist()
                })
            feature_maps_list.append({
                'sample_idx': sample_idx,
                'filters': filters
            })

    return {
        'sample_images': sample_images,
        'feature_maps': feature_maps_list
    }
