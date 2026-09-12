"""Data preparation utilities for AdaBoost (Adaptive Boosting).

This module handles dataset loading, preprocessing, and preparation
for training AdaBoost models.
"""

from typing import Dict, Any, Tuple
import numpy as np
from utils.datasets import DatasetManager


def prepare_data(
    dataset_name: str = 'iris',
    normalize: bool = True,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare dataset for AdaBoost training.

    Loads the specified dataset, optionally normalizes features, and returns
    train/test splits along with metadata for visualization.

    Args:
        dataset_name: Name of the dataset to use ('iris', 'wine', 'digits')
        normalize: Whether to normalize features using StandardScaler
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
            - metadata: Dataset metadata (feature names, target names, etc.)

    Raises:
        ValueError: If dataset_name is not supported for classification
    """
    # Map dataset names to DatasetManager methods
    dataset_loaders = {
        'iris': DatasetManager.get_iris,
        'wine': DatasetManager.get_wine,
        'digits': DatasetManager.get_digits
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported for AdaBoost. "
            f"Supported datasets: {', '.join(dataset_loaders.keys())}"
        )

    # Load dataset
    dataset = dataset_loaders[dataset_name](
        test_size=test_size,
        random_state=random_state
    )

    X_train = dataset['X_train']
    X_test = dataset['X_test']
    y_train = dataset['y_train']
    y_test = dataset['y_test']

    # Normalize if requested
    if normalize:
        X_train, X_test = DatasetManager.normalize_data(X_train, X_test)

    # Prepare metadata for visualization
    metadata = {
        'feature_names': dataset.get('feature_names', []),
        'target_names': dataset.get('target_names', []),
        'description': dataset.get('description', ''),
        'n_samples_train': X_train.shape[0],
        'n_samples_test': X_test.shape[0],
        'n_features': X_train.shape[1],
        'n_classes': len(np.unique(y_train))
    }

    return X_train, X_test, y_train, y_test, metadata


def prepare_visualization_data(
    feature_importance: np.ndarray,
    feature_names: list,
    confusion_matrix_data: np.ndarray,
    class_labels: list,
    learning_curve_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare data for frontend visualization.

    Formats AdaBoost results into a structure suitable for visualization
    components including learning curves, feature importance, and confusion matrices.

    Args:
        feature_importance: Feature importance scores from the ensemble
        feature_names: Names of the features
        confusion_matrix_data: Confusion matrix array
        class_labels: Class labels
        learning_curve_data: Learning curve data across boosting rounds

    Returns:
        Dictionary containing:
            - learning_curve: Learning curve data
            - feature_importance: Feature importance scores
            - confusion_matrix: Confusion matrix data
            - class_labels: Class label names
    """
    # Prepare feature importance data
    feature_importance_dict = {
        name: float(importance)
        for name, importance in zip(feature_names, feature_importance)
    }

    # Sort feature importance by value (descending)
    feature_importance_dict = dict(
        sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # Prepare confusion matrix
    confusion_matrix_list = confusion_matrix_data.tolist()

    # Prepare class labels (convert to strings if needed)
    class_labels_str = [str(label) for label in class_labels]

    return {
        'learning_curve': learning_curve_data,
        'feature_importance': feature_importance_dict,
        'confusion_matrix': confusion_matrix_list,
        'class_labels': class_labels_str
    }
