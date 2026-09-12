"""Data preparation utilities for Logistic Regression.

This module handles dataset loading, preprocessing, and preparation
for training logistic regression models.
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
    """Prepare dataset for logistic regression training.

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
            f"Dataset '{dataset_name}' not supported for Logistic Regression. "
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
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: list,
    feature_names: list = None
) -> Dict[str, Any]:
    """Prepare data for frontend visualization.

    Formats prediction results into a structure suitable for visualization
    components including scatter plots and confusion matrices.

    Args:
        X_test: Test features
        y_test: True test labels
        y_pred: Predicted labels
        probabilities: Predicted class probabilities
        classes: Class labels
        feature_names: Optional feature names for labeling

    Returns:
        Dictionary containing:
            - scatter_data: Data for 2D scatter plot (using first 2 features)
            - confusion_matrix: Confusion matrix data
            - probability_data: Prediction probabilities per sample
            - feature_importance: Feature coefficients (if available)
    """
    from sklearn.metrics import confusion_matrix

    # Prepare scatter plot data (use first 2 features for 2D visualization)
    scatter_data = {
        'x': X_test[:, 0].tolist() if X_test.shape[1] >= 1 else [],
        'y': X_test[:, 1].tolist() if X_test.shape[1] >= 2 else [],
        'true_labels': y_test.tolist(),
        'predicted_labels': y_pred.tolist(),
        'classes': classes,
        'feature_names': feature_names[:2] if feature_names and len(feature_names) >= 2 else ['Feature 1', 'Feature 2']
    }

    # Prepare confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrix_data = {
        'matrix': cm.tolist(),
        'labels': [str(c) for c in classes],
        'class_names': classes if isinstance(classes[0], str) else [f'Class {c}' for c in classes]
    }

    # Prepare probability data (top 5 samples with highest confidence)
    confidence_scores = np.max(probabilities, axis=1)
    top_indices = np.argsort(confidence_scores)[-5:][::-1]

    probability_data = {
        'samples': [
            {
                'index': int(idx),
                'true_label': int(y_test[idx]),
                'predicted_label': int(y_pred[idx]),
                'probabilities': {
                    str(cls): float(prob)
                    for cls, prob in zip(classes, probabilities[idx])
                }
            }
            for idx in top_indices
        ]
    }

    return {
        'scatter_data': scatter_data,
        'confusion_matrix': confusion_matrix_data,
        'probability_data': probability_data
    }
