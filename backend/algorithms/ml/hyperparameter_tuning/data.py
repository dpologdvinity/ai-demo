"""Data loading and preprocessing for Hyperparameter Tuning."""

from typing import Dict, Any
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(
    dataset_name: str = "iris",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """Load and prepare dataset for hyperparameter tuning.

    Args:
        dataset_name: Name of dataset to load (iris, wine, breast_cancer).
        test_size: Proportion of data for test set.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
            - feature_names: List of feature names
            - target_names: List of target class names
            - n_classes: Number of classes
            - n_features: Number of features

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    # Load dataset
    if dataset_name == "iris":
        data = load_iris()
    elif dataset_name == "wine":
        data = load_wine()
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names.tolist() if hasattr(data.target_names, 'tolist') else list(data.target_names)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(feature_names),
        'target_names': target_names,
        'n_classes': len(np.unique(y)),
        'n_features': X.shape[1]
    }


def get_dataset_info(dataset_name: str = "iris") -> Dict[str, Any]:
    """Get metadata about a dataset.

    Args:
        dataset_name: Name of dataset.

    Returns:
        Dictionary with dataset metadata.
    """
    datasets = {
        'iris': {
            'name': 'Iris',
            'n_samples': 150,
            'n_features': 4,
            'n_classes': 3,
            'description': 'Classic iris flower classification dataset with 3 species',
            'task': 'multiclass classification'
        },
        'wine': {
            'name': 'Wine',
            'n_samples': 178,
            'n_features': 13,
            'n_classes': 3,
            'description': 'Wine recognition dataset with chemical analysis',
            'task': 'multiclass classification'
        },
        'breast_cancer': {
            'name': 'Breast Cancer',
            'n_samples': 569,
            'n_features': 30,
            'n_classes': 2,
            'description': 'Breast cancer diagnosis dataset (malignant vs benign)',
            'task': 'binary classification'
        }
    }

    return datasets.get(dataset_name, datasets['iris'])
