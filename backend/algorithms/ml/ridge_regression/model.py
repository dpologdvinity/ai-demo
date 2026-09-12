"""Ridge Regression model implementation using scikit-learn.

This module provides a RidgeRegressionModel class that wraps scikit-learn's
Ridge for predicting continuous values with L2 regularization to prevent overfitting.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time


class RidgeRegressionModel:
    """Ridge Regression implementation for regularized supervised learning.

    This class provides training and prediction functionality for Ridge Regression,
    a supervised learning algorithm that extends linear regression by adding L2
    regularization. The regularization helps prevent overfitting by penalizing
    large coefficient values, making it especially useful for datasets with
    multicollinearity or high dimensionality.

    Attributes:
        model: Scikit-learn Ridge instance
        alpha: Regularization strength parameter
        fit_intercept: Whether to calculate the intercept
        solver: Solver algorithm to use
        training_time_ms: Time taken to train the model in milliseconds
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = 'auto'
    ):
        """Initialize Ridge Regression model.

        Args:
            alpha: Regularization strength. Must be a positive float.
                Larger values specify stronger regularization. Default is 1.0.
            fit_intercept: Whether to calculate the intercept for this model.
                If False, no intercept will be used in calculations (the data
                is expected to be centered). Default is True.
            solver: Solver to use in the computational routines.
                Options: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'.
                Default is 'auto'.
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.training_time_ms = 0.0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Train the Ridge Regression model.

        Fits the ridge regression model to the training data using the specified
        solver and regularization strength.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training target values, shape (n_samples,)

        Returns:
            Dictionary containing:
                - coefficients: Model coefficients (weights)
                - intercept: Model intercept (bias term)
                - n_features: Number of input features
                - alpha: Regularization strength used
                - training_time_ms: Time taken to train in milliseconds

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

        return {
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_) if self.fit_intercept else 0.0,
            "n_features": X_train.shape[1],
            "alpha": self.alpha,
            "training_time_ms": self.training_time_ms
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

        Computes multiple regression metrics to assess model performance,
        including metrics that show the benefit of regularization.

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

        Returns coefficient information and regularization details that can be
        used to understand the impact of L2 regularization on the model.

        Returns:
            Dictionary containing model configuration and learned parameters

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model must be trained to get model info")

        # Calculate L2 norm of coefficients to show regularization effect
        coef_l2_norm = float(np.linalg.norm(self.model.coef_))

        return {
            "fit_intercept": self.fit_intercept,
            "alpha": self.alpha,
            "solver": self.solver,
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_) if self.fit_intercept else 0.0,
            "n_features_in": int(self.model.n_features_in_),
            "coef_l2_norm": coef_l2_norm
        }

    def get_coefficient_analysis(self, feature_names: list = None) -> Dict[str, Any]:
        """Analyze coefficient magnitudes for understanding regularization impact.

        Provides detailed analysis of coefficient values, useful for understanding
        which features are most important and how regularization affects them.

        Args:
            feature_names: Optional list of feature names for labeling

        Returns:
            Dictionary containing:
                - coefficients_abs: Absolute values of coefficients
                - max_coefficient: Feature with largest absolute coefficient
                - min_coefficient: Feature with smallest absolute coefficient
                - mean_coefficient_abs: Mean of absolute coefficient values

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model must be trained to analyze coefficients")

        coefficients = self.model.coef_
        abs_coefficients = np.abs(coefficients)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]

        if len(feature_names) != len(coefficients):
            raise ValueError(
                f"Length of feature_names ({len(feature_names)}) must match "
                f"number of coefficients ({len(coefficients)})"
            )

        # Create coefficient mapping
        coef_map = {name: float(coef) for name, coef in zip(feature_names, coefficients)}
        abs_coef_map = {name: float(abs_coef) for name, abs_coef in zip(feature_names, abs_coefficients)}

        max_idx = np.argmax(abs_coefficients)
        min_idx = np.argmin(abs_coefficients)

        return {
            "coefficients": coef_map,
            "coefficients_abs": abs_coef_map,
            "max_coefficient": {
                "feature": feature_names[max_idx],
                "value": float(coefficients[max_idx])
            },
            "min_coefficient": {
                "feature": feature_names[min_idx],
                "value": float(coefficients[min_idx])
            },
            "mean_coefficient_abs": float(np.mean(abs_coefficients))
        }
