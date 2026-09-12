"""Elastic Net Regression model implementation using scikit-learn.

This module provides an ElasticNetModel class that wraps scikit-learn's
ElasticNet for predicting continuous values with combined L1 and L2 regularization.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time


class ElasticNetModel:
    """Elastic Net Regression implementation with combined L1 and L2 regularization.

    This class provides training and prediction functionality for Elastic Net Regression,
    a linear model that linearly combines the L1 and L2 penalties of Lasso and Ridge.
    This is useful when there are multiple correlated features, as Lasso tends to select
    only one feature from a group while Elastic Net selects all of them with distributed
    weights.

    Attributes:
        model: Scikit-learn ElasticNet instance
        alpha: Overall regularization strength
        l1_ratio: Mix ratio between L1 and L2 regularization
        max_iter: Maximum number of iterations for optimization
        fit_intercept: Whether to calculate the intercept
        training_time_ms: Time taken to train the model in milliseconds
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        fit_intercept: bool = True,
        random_state: int = 42
    ):
        """Initialize Elastic Net Regression model.

        Args:
            alpha: Overall regularization strength. Higher values create more
                regularization. Must be positive.
            l1_ratio: The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
                For l1_ratio = 0 the penalty is an L2 penalty (Ridge).
                For l1_ratio = 1 it is an L1 penalty (Lasso).
                For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
            max_iter: Maximum number of iterations for the optimization algorithm.
            fit_intercept: Whether to calculate the intercept for this model.
            random_state: Random seed for reproducibility.
        """
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            fit_intercept=fit_intercept,
            random_state=random_state
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.training_time_ms = 0.0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Train the Elastic Net Regression model.

        Fits the Elastic Net model to the training data using coordinate descent
        with combined L1 and L2 regularization.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training target values, shape (n_samples,)

        Returns:
            Dictionary containing:
                - coefficients: Model coefficients (weights)
                - intercept: Model intercept (bias term)
                - n_features: Number of input features
                - n_nonzero_coefs: Number of non-zero coefficients
                - sparsity: Percentage of coefficients that are exactly zero
                - l2_norm: L2 norm of coefficients showing regularization effect
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
        n_nonzero = np.sum(np.abs(self.model.coef_) > 1e-10)
        n_total = len(self.model.coef_)
        sparsity = (1 - n_nonzero / n_total) * 100

        # Calculate L2 norm to show regularization effect
        l2_norm = float(np.linalg.norm(self.model.coef_))

        return {
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
            "n_features": X_train.shape[1],
            "n_nonzero_coefs": int(n_nonzero),
            "sparsity": float(sparsity),
            "l2_norm": l2_norm,
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
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "fit_intercept": self.fit_intercept,
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
            "n_features_in": int(self.model.n_features_in_),
            "n_iter": int(self.model.n_iter_)
        }

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """Get feature importance based on absolute coefficient values.

        Features with coefficients close to zero have low importance.

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

    def compare_regularization_types(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list
    ) -> Dict[str, Any]:
        """Train models with different l1_ratio values to show regularization effects.

        This demonstrates how Elastic Net transitions from Ridge (l1_ratio=0) to
        Lasso (l1_ratio=1).

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            feature_names: List of feature names

        Returns:
            Dictionary containing comparison data for visualization
        """
        l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        comparison_data = []

        for l1_ratio in l1_ratios:
            temp_model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=l1_ratio,
                max_iter=self.max_iter,
                fit_intercept=self.fit_intercept,
                random_state=42
            )
            temp_model.fit(X_train, y_train)

            y_pred = temp_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            n_nonzero = np.sum(np.abs(temp_model.coef_) > 1e-10)
            sparsity = (1 - n_nonzero / len(temp_model.coef_)) * 100
            l2_norm = float(np.linalg.norm(temp_model.coef_))

            comparison_data.append({
                "l1_ratio": l1_ratio,
                "r2_score": float(r2),
                "n_nonzero_coefs": int(n_nonzero),
                "sparsity": float(sparsity),
                "l2_norm": l2_norm,
                "coefficients": temp_model.coef_.tolist()
            })

        return {
            "comparison_data": comparison_data,
            "feature_names": feature_names
        }
