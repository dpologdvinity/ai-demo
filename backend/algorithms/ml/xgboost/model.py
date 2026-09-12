"""XGBoost (Gradient Boosting) classifier implementation.

This module implements XGBoost, an optimized gradient boosting framework that
builds an ensemble of decision trees sequentially to minimize prediction errors.
"""

import time
from typing import Dict, Any, List
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from utils.datasets import DatasetManager


class XGBoostModel:
    """XGBoost classifier for multi-class classification.

    XGBoost (eXtreme Gradient Boosting) is an ensemble learning method that
    builds trees sequentially, where each tree tries to correct the errors
    made by previous trees. It uses gradient descent optimization to minimize
    the loss function.

    Key Features:
        - Sequential tree building with error correction
        - Regularization to prevent overfitting
        - Parallel tree construction
        - Handles missing values automatically
        - Built-in cross-validation

    Attributes:
        model: The XGBClassifier instance.
        feature_names: Names of input features.
        target_names: Names of target classes.
    """

    def __init__(self):
        """Initialize XGBoost model."""
        self.model = None
        self.feature_names = None
        self.target_names = None

    def train(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 1.0,
        dataset_name: str = "wine",
        normalize: bool = True
    ) -> Dict[str, Any]:
        """Train XGBoost classifier with specified parameters.

        Args:
            n_estimators: Number of boosting rounds (trees to build).
            learning_rate: Step size shrinkage to prevent overfitting.
            max_depth: Maximum tree depth for base learners.
            subsample: Subsample ratio of training instances.
            dataset_name: Name of dataset to use.
            normalize: Whether to normalize features.

        Returns:
            Dictionary containing:
                - success: Whether training completed successfully
                - metrics: Performance metrics
                - predictions: Test set predictions
                - visualization_data: Data for charts
                - execution_time_ms: Training time in milliseconds
                - parameters_used: Actual parameters used

        Raises:
            ValueError: If parameters are invalid or dataset cannot be loaded.
            RuntimeError: If training fails.
        """
        start_time = time.time()

        try:
            # Load dataset
            if dataset_name == "wine":
                dataset = DatasetManager.get_wine()
            elif dataset_name == "iris":
                dataset = DatasetManager.get_iris()
            elif dataset_name == "digits":
                dataset = DatasetManager.get_digits()
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            X_train = dataset['X_train']
            X_test = dataset['X_test']
            y_train = dataset['y_train']
            y_test = dataset['y_test']
            self.feature_names = dataset.get('feature_names', [])
            self.target_names = dataset.get('target_names', [])

            # Normalize data if requested
            if normalize:
                X_train, X_test = DatasetManager.normalize_data(X_train, X_test)

            # Initialize and train XGBoost model
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42,
                objective='multi:softmax',
                num_class=len(np.unique(y_train)),
                eval_metric='mlogloss'
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Get feature importance
            feature_importance = self.model.feature_importances_.tolist()

            # Generate learning curves (track training progress)
            learning_curves = self._generate_learning_curves(
                X_train, y_train, X_test, y_test,
                n_estimators, learning_rate, max_depth, subsample
            )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            return {
                'success': True,
                'metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                },
                'predictions': y_pred.tolist(),
                'visualization_data': {
                    'confusion_matrix': conf_matrix.tolist(),
                    'feature_importance': {
                        'features': self.feature_names if self.feature_names else
                                   [f"Feature {i}" for i in range(len(feature_importance))],
                        'importance': feature_importance
                    },
                    'learning_curves': learning_curves,
                    'class_probabilities': y_pred_proba.tolist(),
                    'target_names': self.target_names if self.target_names else
                                   [f"Class {i}" for i in range(len(np.unique(y_test)))]
                },
                'execution_time_ms': execution_time_ms,
                'parameters_used': {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'subsample': subsample,
                    'dataset_name': dataset_name,
                    'normalize': normalize
                },
                'error': None
            }

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return {
                'success': False,
                'metrics': {},
                'predictions': None,
                'visualization_data': {},
                'execution_time_ms': execution_time_ms,
                'parameters_used': {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'subsample': subsample,
                    'dataset_name': dataset_name,
                    'normalize': normalize
                },
                'error': str(e)
            }

    def _generate_learning_curves(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_estimators: int,
        learning_rate: float,
        max_depth: int,
        subsample: float
    ) -> Dict[str, List[float]]:
        """Generate learning curves by training with increasing numbers of estimators.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            n_estimators: Maximum number of estimators.
            learning_rate: Learning rate for boosting.
            max_depth: Maximum tree depth.
            subsample: Subsample ratio.

        Returns:
            Dictionary with train/test accuracy curves.
        """
        # Sample at most 10 points along the learning curve
        step = max(1, n_estimators // 10)
        estimator_range = list(range(step, n_estimators + 1, step))
        if estimator_range[-1] != n_estimators:
            estimator_range.append(n_estimators)

        train_scores = []
        test_scores = []

        for n in estimator_range:
            model = XGBClassifier(
                n_estimators=n,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42,
                objective='multi:softmax',
                num_class=len(np.unique(y_train)),
                eval_metric='mlogloss'
            )
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_scores.append(float(accuracy_score(y_train, train_pred)))
            test_scores.append(float(accuracy_score(y_test, test_pred)))

        return {
            'n_estimators': estimator_range,
            'train_accuracy': train_scores,
            'test_accuracy': test_scores
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Input features.

        Returns:
            Predicted class labels.

        Raises:
            RuntimeError: If model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input features.

        Returns:
            Predicted class probabilities.

        Raises:
            RuntimeError: If model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet")
        return self.model.predict_proba(X)
