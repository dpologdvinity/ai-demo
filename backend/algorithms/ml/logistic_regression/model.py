"""Logistic Regression algorithm implementation.

This module implements a Logistic Regression classifier using scikit-learn
for binary and multiclass classification tasks.
"""

import time
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LogisticRegressionModel:
    """Logistic Regression classifier wrapper.

    This class provides a clean interface for training and evaluating
    logistic regression models with configurable hyperparameters.

    Attributes:
        C: Regularization strength (inverse of regularization parameter)
        penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
        max_iter: Maximum number of iterations for optimization
        solver: Optimization algorithm to use
        model: Trained scikit-learn LogisticRegression instance
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = 'l2',
        max_iter: int = 100,
        solver: str = 'lbfgs',
        random_state: int = 42
    ):
        """Initialize Logistic Regression model with hyperparameters.

        Args:
            C: Regularization strength (smaller values = stronger regularization)
            penalty: Type of regularization ('l1', 'l2', 'elasticnet', 'none')
            max_iter: Maximum iterations for solver convergence
            solver: Optimization algorithm ('lbfgs', 'liblinear', 'saga', etc.)
            random_state: Random seed for reproducibility
        """
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state

        # Adjust solver based on penalty if needed
        if penalty == 'l1' and solver == 'lbfgs':
            # lbfgs doesn't support l1, use saga or liblinear
            self.solver = 'saga'

        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            max_iter=max_iter,
            solver=self.solver,
            random_state=random_state,
            multi_class='auto'
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train the logistic regression model and evaluate performance.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing:
                - metrics: Performance metrics (accuracy, precision, recall, f1)
                - predictions: Predictions on test set
                - probabilities: Predicted class probabilities
                - execution_time_ms: Training time in milliseconds
                - coefficients: Model coefficients
                - intercept: Model intercept
        """
        start_time = time.time()

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        # Calculate metrics
        # For multiclass, use weighted average
        average = 'binary' if len(np.unique(y_train)) == 2 else 'weighted'

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average=average, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average=average, zero_division=0))
        }

        return {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist(),
            'execution_time_ms': execution_time,
            'coefficients': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist(),
            'classes': self.model.classes_.tolist()
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Predicted class labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities for new data.

        Args:
            X: Features to predict on

        Returns:
            Predicted class probabilities
        """
        return self.model.predict_proba(X)
