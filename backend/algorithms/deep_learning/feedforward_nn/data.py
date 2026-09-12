"""Data loading and preprocessing for Feedforward Neural Network (MLP)."""

from typing import Dict, Any
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_iris_data(
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True
) -> Dict[str, Any]:
    """Load and preprocess Iris dataset for MLP.

    The Iris dataset contains measurements of iris flowers with 4 features
    (sepal length, sepal width, petal length, petal width) and 3 classes
    (setosa, versicolor, virginica).

    Args:
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        normalize: Whether to normalize features using StandardScaler

    Returns:
        Dictionary containing:
            - X_train: Training features (N, 4)
            - X_test: Test features (M, 4)
            - y_train: Training labels
            - y_test: Test labels
            - feature_names: List of feature names
            - target_names: List of class names
            - num_classes: Number of classes
            - scaler: StandardScaler instance (if normalize=True)
    """
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalize features
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist(),
        'num_classes': len(iris.target_names),
        'scaler': scaler
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the Iris dataset.

    Returns:
        Dictionary with dataset metadata
    """
    iris = load_iris()

    return {
        'name': 'Iris',
        'description': 'Classic iris flower classification dataset',
        'num_samples': len(iris.data),
        'num_features': len(iris.feature_names),
        'num_classes': len(iris.target_names),
        'features': iris.feature_names,
        'classes': iris.target_names.tolist(),
        'class_distribution': {
            iris.target_names[i]: int(np.sum(iris.target == i))
            for i in range(len(iris.target_names))
        },
        'use_case': 'Multi-class classification with continuous features'
    }


def prepare_visualization_data(
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list
) -> Dict[str, Any]:
    """Prepare visualization data for frontend.

    Args:
        X_sample: Sample features
        y_sample: Sample actual labels
        y_pred: Sample predicted labels
        feature_names: Names of features

    Returns:
        Dictionary with visualization data
    """
    # Convert samples to list format
    sample_data = []
    for i in range(min(10, len(X_sample))):
        sample_data.append({
            'features': {
                feature_names[j]: float(X_sample[i][j])
                for j in range(len(feature_names))
            },
            'actual': int(y_sample[i]),
            'predicted': int(y_pred[i]),
            'correct': int(y_sample[i]) == int(y_pred[i])
        })

    return {
        'sample_predictions': sample_data
    }
