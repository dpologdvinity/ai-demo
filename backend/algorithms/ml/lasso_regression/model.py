"""Lasso Regression model implementation using scikit-learn.

This module provides a LassoRegressionModel class that wraps scikit-learn's
Lasso for predicting continuous values with L1 regularization for feature selection.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time


class LassoRegressionModel:
    """Lasso Regression implementation with L1 regularization.

    This class provides training and prediction functionality for Lasso Regression,
    a linear model with L1 regularization that performs both prediction and automatic
    feature selection by driving some coefficients to exactly zero.

    Attributes:
        model: Scikit-learn Lasso instance
        alpha: Regularization strength (L1 penalty)
        max_iter: Maximum number of iterations for optimization
        selection: Coefficient update rule ('cyclic' or 'random')
        training_time_ms: Time taken to train the model in milliseconds
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        selection: str = 'cyclic',
        random_state: int = 42
    ):
        """Initialize Lasso Regression model.

        Args:
            alpha: Regularization strength. Higher values create sparser models
                with more coefficients set to zero. Must be positive.
            max_iter: Maximum number of iterations for the optimization algorithm.
            selection: Strategy for updating coefficients:
                - 'cyclic': Updates coefficients in sequential order
                - 'random': Updates coefficients in random order (faster for large datasets)
            random_state: Random seed for reproducibility when selection='random'.
        """
        self.model = Lasso(
            alpha=alpha,
            max_iter=max_iter,
            selection=selection,
            random_state=random_state
        )
        self.alpha = alpha
        self.max_iter = max_iter
        self.selection = selection
        self.training_time_ms = 0.0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Train the Lasso Regression model.

        Fits the Lasso model to the training data using coordinate descent
        with L1 regularization for automatic feature selection.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training target values, shape (n_samples,)

        Returns:
            Dictionary containing:
                - coefficients: Model coefficients (weights) - some may be exactly zero
                - intercept: Model intercept (bias term)
                - n_features: Number of input features
                - n_nonzero_coefs: Number of non-zero coefficients (selected features)
                - sparsity: Percentage of coefficients that are exactly zero
                - training_time_ms: Time taken to train in milliseconds
                - converged: Whether the optimization converged

        Raises:
            ValueError: If X_train or y_train are empty or have incompatible shapes
        """
        if X_train.size == 0:
            raise ValueError("X_train cannot be empty")
        if y_train.size == 0:
            raise ValueError("y_train cannot be empty")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples, "
                f"got {X_train.shape[0]} and {y_train.shape[0]}"
            )

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time_ms = (time.time() - start_time) * 1000

        # Count non-zero coefficients to show sparsity
        n_nonzero = np.sum(self.model.coef_ != 0)
        n_total = len(self.model.coef_)
        sparsity = (1 - n_nonzero / n_total) * 100

        return {
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
            "n_features": X_train.shape[1],
            "n_nonzero_coefs": int(n_nonzero),
            "sparsity": float(sparsity),
            "training_time_ms": self.training_time_ms,
            "converged": bool(self.model.n_iter_ < self.max_iter)
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.

        Args:
            X: Features to predict on, shape (n_samples, n_features)

        Returns:
            Predicted values, shape (n_samples,)

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance on test data.

        Computes multiple regression metrics to assess model performance.

        Args:
            X_test: Test features, shape (n_samples, n_features)
            y_test: True target values, shape (n_samples,)

        Returns:
            Dictionary containing:
                - r2_score: R² (coefficient of determination) score
                - mse: Mean Squared Error
                - rmse: Root Mean Squared Error
                - mae: Mean Absolute Error

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "r2_score": float(r2),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and learned parameters.

        Returns:
            Dictionary containing model configuration and learned parameters

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model must be trained to get model info")

        return {
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "selection": self.selection,
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
            "n_features_in": int(self.model.n_features_in_),
            "n_iter": int(self.model.n_iter_)
        }

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """Get feature importance based on absolute coefficient values.

        Features with zero coefficients were not selected by Lasso.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to absolute coefficient values

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model must be trained to get feature importance")

        return {
            name: abs(float(coef))
            for name, coef in zip(feature_names, self.model.coef_)
        }
