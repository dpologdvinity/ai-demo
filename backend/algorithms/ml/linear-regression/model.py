"""Linear Regression model implementation using scikit-learn.

This module provides a LinearRegressionModel class that wraps scikit-learn's
LinearRegression for predicting continuous values using linear relationships.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time


class LinearRegressionModel:
    """Linear Regression implementation for supervised learning.

    This class provides training and prediction functionality for Linear Regression,
    a fundamental supervised learning algorithm that models the relationship between
    features and a continuous target variable using a linear equation.

    Attributes:
        model: Scikit-learn LinearRegression instance
        fit_intercept: Whether to calculate the intercept
        training_time_ms: Time taken to train the model in milliseconds
    """

    def __init__(self, fit_intercept: bool = True):
        """Initialize Linear Regression model.

        Args:
            fit_intercept: Whether to calculate the intercept for this model.
                If False, no intercept will be used in calculations (the data
                is expected to be centered).
        """
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.fit_intercept = fit_intercept
        self.training_time_ms = 0.0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Train the Linear Regression model.

        Fits the linear model to the training data using ordinary least squares.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training target values, shape (n_samples,)

        Returns:
            Dictionary containing:
                - coefficients: Model coefficients (weights)
                - intercept: Model intercept (bias term)
                - n_features: Number of input features
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
            "fit_intercept": self.fit_intercept,
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_) if self.fit_intercept else 0.0,
            "n_features_in": int(self.model.n_features_in_),
        }
