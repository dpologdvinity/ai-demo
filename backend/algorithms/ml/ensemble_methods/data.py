"""Data preparation utilities for Ensemble Methods.

This module handles dataset loading, preprocessing, and preparation
for training ensemble models.
"""

from typing import Dict, Any, Tuple
import numpy as np
from utils.datasets import DatasetManager


def prepare_data(
    dataset_name: str = 'wine',
    normalize: bool = True,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare dataset for Ensemble Methods training.

    Loads the specified dataset, optionally normalizes features, and returns
    train/test splits along with metadata for visualization.

    Args:
        dataset_name: Name of the dataset to use ('wine', 'breast_cancer', 'iris')
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
        'wine': DatasetManager.get_wine,
        'breast_cancer': DatasetManager.get_breast_cancer,
        'iris': DatasetManager.get_iris,
        'digits': DatasetManager.get_digits
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported for Ensemble Methods. "
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
    performance_comparison: Dict[str, Any],
    feature_importance: np.ndarray,
    feature_names: list,
    confusion_matrix_data: np.ndarray,
    class_labels: list,
    diversity_metrics: Dict[str, float],
    voting_data: Dict[str, Any],
    individual_predictions: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Prepare data for frontend visualization.

    Formats ensemble results into a structure suitable for visualization
    components including performance comparison, diversity metrics, and voting patterns.

    Args:
        performance_comparison: Comparison of single model vs ensemble performance
        feature_importance: Feature importance scores from the ensemble
        feature_names: Names of the features
        confusion_matrix_data: Confusion matrix array
        class_labels: Class labels
        diversity_metrics: Diversity metrics between models
        voting_data: Voting patterns and agreement data
        individual_predictions: Predictions from each model

    Returns:
        Dictionary containing formatted visualization data
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

    # Convert individual predictions to lists
    individual_preds_dict = {
        name: preds.tolist() if isinstance(preds, np.ndarray) else preds
        for name, preds in individual_predictions.items()
    }

    return {
        'performance_comparison': performance_comparison,
        'feature_importance': feature_importance_dict,
        'confusion_matrix': confusion_matrix_list,
        'class_labels': class_labels_str,
        'diversity_metrics': diversity_metrics,
        'voting_patterns': voting_data,
        'individual_predictions': individual_preds_dict
    }
