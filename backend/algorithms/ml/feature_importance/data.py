"""Data loading utilities for Feature Importance Analysis."""

from typing import Dict, Any, List
import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine
from sklearn.model_selection import train_test_split


def load_dataset(dataset_name: str = 'housing', test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """Load a dataset for feature importance analysis.

    Args:
        dataset_name: Name of dataset ('housing', 'diabetes', 'wine')
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training target
            - y_test: Test target
            - feature_names: List of feature names
            - target_name: Name of target variable
            - dataset_info: Metadata about the dataset
            - is_classification: Whether it's a classification task

    Raises:
        ValueError: If dataset name is not supported
    """
    if dataset_name == 'housing':
        data = fetch_california_housing()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_name = 'MedHouseVal'
        is_classification = False
        description = 'California Housing Dataset - Predict median house values'
        n_samples, n_features = X.shape

    elif dataset_name == 'diabetes':
        data = load_diabetes()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_name = 'Disease Progression'
        is_classification = False
        description = 'Diabetes Dataset - Predict disease progression'
        n_samples, n_features = X.shape

    elif dataset_name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_name = 'Wine Class'
        is_classification = True
        description = 'Wine Dataset - Classify wine types'
        n_samples, n_features = X.shape

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from 'housing', 'diabetes', or 'wine'")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    dataset_info = {
        'name': dataset_name,
        'description': description,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'target_name': target_name,
        'is_classification': is_classification
    }

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(feature_names),
        'target_name': target_name,
        'dataset_info': dataset_info,
        'is_classification': is_classification
    }


def get_dataset_info(dataset_name: str = 'housing') -> Dict[str, Any]:
    """Get metadata about a dataset without loading it.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary containing dataset metadata
    """
    datasets = {
        'housing': {
            'name': 'California Housing',
            'description': 'Predict median house values based on California census data',
            'n_samples': 20640,
            'n_features': 8,
            'target': 'Median House Value ($100k)',
            'type': 'regression',
            'features': [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude'
            ]
        },
        'diabetes': {
            'name': 'Diabetes Dataset',
            'description': 'Predict disease progression from baseline medical measurements',
            'n_samples': 442,
            'n_features': 10,
            'target': 'Disease Progression',
            'type': 'regression',
            'features': [
                'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'
            ]
        },
        'wine': {
            'name': 'Wine Recognition Dataset',
            'description': 'Classify wines into 3 classes based on chemical analysis',
            'n_samples': 178,
            'n_features': 13,
            'target': 'Wine Class',
            'type': 'classification',
            'features': [
                'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
                'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
                'proline'
            ]
        }
    }

    return datasets.get(dataset_name, datasets['housing'])


def get_available_datasets() -> List[str]:
    """Get list of available datasets.

    Returns:
        List of dataset names
    """
    return ['housing', 'diabetes', 'wine']
