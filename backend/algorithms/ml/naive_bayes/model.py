"""Naive Bayes algorithm implementation using scikit-learn.

This module implements a Gaussian Naive Bayes classifier for probabilistic
classification tasks based on Bayes' theorem with feature independence assumption.
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from backend.utils.datasets import DatasetManager


class NaiveBayesModel:
    """Gaussian Naive Bayes classifier implementation.

    Implements a probabilistic classifier based on applying Bayes' theorem
    with strong (naive) independence assumptions between features. Uses
    Gaussian distribution assumption for continuous features.

    Attributes:
        model: The scikit-learn GaussianNB model instance
        var_smoothing: Variance smoothing parameter
        priors: Prior probabilities of classes
        classes_: The class labels (available after training)
        class_priors_: The learned prior probabilities (available after training)

    Example:
        >>> nb_model = NaiveBayesModel(var_smoothing=1e-9)
        >>> result = nb_model.train(X_train, y_train, X_test, y_test)
        >>> print(f"Accuracy: {result['metrics']['accuracy']:.3f}")
    """

    def __init__(
        self,
        var_smoothing: float = 1e-9,
        priors: Optional[List[float]] = None
    ):
        """Initialize Naive Bayes model.

        Args:
            var_smoothing: Portion of the largest variance of all features
                that is added to variances for calculation stability.
                Must be between 1e-10 and 1e-8.
            priors: Prior probabilities of the classes. If specified,
                must sum to 1. If None, priors are adjusted according to the data.

        Raises:
            ValueError: If var_smoothing is out of valid range or priors don't sum to 1.
        """
        if not 1e-10 <= var_smoothing <= 1e-8:
            raise ValueError(
                f"var_smoothing must be between 1e-10 and 1e-8, got {var_smoothing}"
            )

        if priors is not None:
            priors_array = np.array(priors)
            if not np.allclose(priors_array.sum(), 1.0):
                raise ValueError(
                    f"Priors must sum to 1, got {priors_array.sum()}"
                )

        self.var_smoothing = var_smoothing
        self.priors = np.array(priors) if priors is not None else None
        self.model = GaussianNB(
            priors=self.priors,
            var_smoothing=var_smoothing
        )
        self.classes_ = None
        self.class_priors_ = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train the Naive Bayes model and evaluate on test data.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
            y_train: Training target vector of shape (n_samples,)
            X_test: Test feature matrix of shape (n_test_samples, n_features)
            y_test: Test target vector of shape (n_test_samples,)
            class_names: Optional list of class names for visualization

        Returns:
            Dictionary containing:
                - metrics: Performance metrics (accuracy, precision, recall, f1)
                - predictions: Model predictions on test set
                - probabilities: Class probability distributions
                - feature_contributions: Contribution of features to predictions
                - visualization_data: Confusion matrix and probability data
                - execution_time_ms: Training time in milliseconds

        Raises:
            ValueError: If input arrays have incompatible shapes or invalid data.
        """
        start_time = time.time()

        # Validate inputs
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples, "
                f"got {X_train.shape[0]} and {y_train.shape[0]}"
            )
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"X_test and y_test must have same number of samples, "
                f"got {X_test.shape[0]} and {y_test.shape[0]}"
            )
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"X_train and X_test must have same number of features, "
                f"got {X_train.shape[1]} and {X_test.shape[1]}"
            )

        # Train the model
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        self.class_priors_ = self.model.class_prior_

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(
                y_test, y_pred, average='weighted', zero_division=0
            )),
            "recall": float(recall_score(
                y_test, y_pred, average='weighted', zero_division=0
            )),
            "f1_score": float(f1_score(
                y_test, y_pred, average='weighted', zero_division=0
            ))
        }

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Calculate feature contributions (log probability contributions)
        feature_contributions = self._calculate_feature_contributions(
            X_test[:10]  # Only compute for first 10 samples for efficiency
        )

        # Prepare visualization data
        visualization_data = {
            "confusion_matrix": conf_matrix.tolist(),
            "class_names": class_names if class_names else [
                f"Class {i}" for i in self.classes_
            ],
            "probability_distributions": {
                "samples": y_prob[:20].tolist(),  # First 20 samples
                "true_labels": y_test[:20].tolist(),
                "predicted_labels": y_pred[:20].tolist()
            },
            "class_priors": self.class_priors_.tolist(),
            "feature_importance": self._calculate_feature_importance(X_train)
        }

        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "metrics": metrics,
            "predictions": y_pred.tolist(),
            "probabilities": y_prob.tolist(),
            "feature_contributions": feature_contributions,
            "visualization_data": visualization_data,
            "execution_time_ms": execution_time
        }

    def _calculate_feature_contributions(
        self,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate contribution of each feature to predictions.

        For Naive Bayes, this computes the log probability contribution
        of each feature for each class.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Dictionary with feature contribution data
        """
        if not hasattr(self.model, 'theta_'):
            return {}

        # Get class means (theta) and variances (var)
        theta = self.model.theta_  # Shape: (n_classes, n_features)
        var = self.model.var_  # Shape: (n_classes, n_features)

        # Calculate log probability contributions for each feature
        contributions = []
        for sample in X:
            sample_contrib = []
            for class_idx in range(len(self.classes_)):
                # Calculate log probability contribution of each feature
                feature_log_probs = -0.5 * np.log(2 * np.pi * var[class_idx, :])
                feature_log_probs -= 0.5 * ((sample - theta[class_idx, :]) ** 2) / var[class_idx, :]
                sample_contrib.append(feature_log_probs.tolist())
            contributions.append(sample_contrib)

        return {
            "log_probability_contributions": contributions,
            "class_means": theta.tolist(),
            "class_variances": var.tolist()
        }

    def _calculate_feature_importance(
        self,
        X_train: np.ndarray
    ) -> Dict[str, List[float]]:
        """Calculate feature importance based on variance ratios.

        Features with higher between-class variance relative to within-class
        variance are more important for classification.

        Args:
            X_train: Training feature matrix

        Returns:
            Dictionary with feature importance scores
        """
        if not hasattr(self.model, 'theta_'):
            return {}

        theta = self.model.theta_  # Class means
        var = self.model.var_  # Class variances

        # Calculate between-class variance
        overall_mean = theta.mean(axis=0)
        between_class_var = np.var(theta, axis=0)

        # Calculate average within-class variance
        within_class_var = var.mean(axis=0)

        # Feature importance: between-class variance / within-class variance
        # Add small epsilon to avoid division by zero
        importance = between_class_var / (within_class_var + 1e-10)

        # Normalize to [0, 1] range
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)

        return {
            "importance_scores": importance.tolist(),
            "between_class_variance": between_class_var.tolist(),
            "within_class_variance": within_class_var.tolist()
        }

    @classmethod
    def from_dataset(
        cls,
        dataset_name: str = "iris",
        var_smoothing: float = 1e-9,
        priors: Optional[List[float]] = None,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """Train Naive Bayes model on a specified dataset.

        Convenience method that loads a dataset, creates a model,
        trains it, and returns results.

        Args:
            dataset_name: Name of dataset to use ('iris', 'wine', 'digits')
            var_smoothing: Variance smoothing parameter
            priors: Prior probabilities of classes
            normalize: Whether to normalize features before training

        Returns:
            Dictionary containing training results and metadata

        Raises:
            ValueError: If dataset_name is not supported or parameters are invalid.
        """
        # Load dataset
        if dataset_name == "iris":
            data = DatasetManager.get_iris()
        elif dataset_name == "wine":
            data = DatasetManager.get_wine()
        elif dataset_name == "digits":
            data = DatasetManager.get_digits()
        else:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported datasets: iris, wine, digits"
            )

        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        # Normalize if requested
        if normalize:
            X_train, X_test = DatasetManager.normalize_data(X_train, X_test)

        # Get class names if available
        class_names = data.get('target_names')

        # Create and train model
        model = cls(var_smoothing=var_smoothing, priors=priors)
        results = model.train(X_train, y_train, X_test, y_test, class_names)

        # Add parameters used
        results['parameters_used'] = {
            'var_smoothing': var_smoothing,
            'priors': priors,
            'dataset_name': dataset_name,
            'normalize': normalize
        }

        return results
