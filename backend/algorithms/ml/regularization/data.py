"""Data loading and preparation for Regularization Techniques."""

from typing import Dict, Any
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def create_overfitting_prone_dataset(
    n_samples: int = 200,
    n_features: int = 100,
    n_informative: int = 20,
    n_redundant: int = 10,
    noise: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    """Create a synthetic overfitting-prone dataset.

    This dataset is intentionally designed to demonstrate overfitting:
    - Small sample size relative to features (200 samples, 100 features)
    - High noise level
    - Many redundant and random features

    This makes it ideal for demonstrating the benefits of regularization.

    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        noise: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - X_train: Training features (numpy array)
            - X_test: Test features (numpy array)
            - y_train: Training target values (numpy array)
            - y_test: Test target values (numpy array)
            - feature_names: List of feature names
            - description: Dataset description
    """
    # Ensure n_informative + n_redundant <= n_features
    # Adjust if necessary
    if n_informative + n_redundant >= n_features:
        n_informative = max(int(n_features * 0.2), 5)
        n_redundant = max(int(n_features * 0.1), 2)

    # Generate high-dimensional classification problem converted to regression
    X, y_class = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        flip_y=noise,
        random_state=random_state
    )

    # Convert to regression problem by using probability scores with added noise
    np.random.seed(random_state)
    y = y_class.astype(float) + np.random.normal(0, noise, size=n_samples)

    # Split into train/test with 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    # Generate feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'description': (
            f'Synthetic overfitting-prone dataset: {n_samples} samples, '
            f'{n_features} features ({n_informative} informative, '
            f'{n_redundant} redundant, noise={noise})'
        )
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the regularization dataset.

    Returns:
        Dictionary containing dataset metadata
    """
    return {
        'name': 'Overfitting-Prone Synthetic Dataset',
        'n_samples': 200,
        'n_features': 100,
        'n_informative': 20,
        'n_redundant': 10,
        'target_description': 'Continuous target with noise',
        'description': (
            'High-dimensional synthetic dataset designed to demonstrate '
            'overfitting. Small sample size (200) with many features (100) '
            'including redundant and noisy features, ideal for showing how '
            'regularization techniques prevent overfitting.'
        )
    }
