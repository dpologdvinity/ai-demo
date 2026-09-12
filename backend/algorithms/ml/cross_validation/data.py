"""
Data loading and preparation for Cross-Validation.

This module provides functions to load various datasets for cross-validation evaluation.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def load_iris_data() -> Dict[str, Any]:
    """Load the Iris dataset.

    Returns:
        Dictionary containing:
            - X: Features array
            - y: Target array
            - feature_names: List of feature names
            - target_names: List of target class names
            - dataset_name: Name of the dataset
    """
    iris = datasets.load_iris()
    return {
        'X': iris.data,
        'y': iris.target,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist(),
        'dataset_name': 'iris'
    }


def load_wine_data() -> Dict[str, Any]:
    """Load the Wine dataset.

    Returns:
        Dictionary containing:
            - X: Features array
            - y: Target array
            - feature_names: List of feature names
            - target_names: List of target class names
            - dataset_name: Name of the dataset
    """
    wine = datasets.load_wine()
    return {
        'X': wine.data,
        'y': wine.target,
        'feature_names': wine.feature_names,
        'target_names': wine.target_names.tolist(),
        'dataset_name': 'wine'
    }


def load_breast_cancer_data() -> Dict[str, Any]:
    """Load the Breast Cancer dataset.

    Returns:
        Dictionary containing:
            - X: Features array
            - y: Target array
            - feature_names: List of feature names
            - target_names: List of target class names
            - dataset_name: Name of the dataset
    """
    cancer = datasets.load_breast_cancer()
    return {
        'X': cancer.data,
        'y': cancer.target,
        'feature_names': cancer.feature_names,
        'target_names': cancer.target_names.tolist(),
        'dataset_name': 'breast_cancer'
    }


def load_dataset(dataset_name: str, normalize: bool = True) -> Dict[str, Any]:
    """Load specified dataset with optional normalization.

    Args:
        dataset_name: Name of dataset ('iris', 'wine', 'breast_cancer')
        normalize: Whether to normalize features

    Returns:
        Dictionary containing dataset information

    Raises:
        ValueError: If dataset_name is not recognized
    """
    if dataset_name == 'iris':
        data = load_iris_data()
    elif dataset_name == 'wine':
        data = load_wine_data()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        data['X'] = scaler.fit_transform(data['X'])

    return data


def get_dataset_info(dataset_name: str = 'iris') -> Dict[str, Any]:
    """Get information about a dataset without loading it.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary with dataset metadata
    """
    info = {
        'iris': {
            'name': 'Iris',
            'n_samples': 150,
            'n_features': 4,
            'n_classes': 3,
            'description': 'Classic iris flower dataset with 3 species'
        },
        'wine': {
            'name': 'Wine',
            'n_samples': 178,
            'n_features': 13,
            'n_classes': 3,
            'description': 'Wine classification dataset with chemical analysis'
        },
        'breast_cancer': {
            'name': 'Breast Cancer',
            'n_samples': 569,
            'n_features': 30,
            'n_classes': 2,
            'description': 'Breast cancer diagnostic dataset (malignant vs benign)'
        }
    }

    return info.get(dataset_name, info['iris'])
