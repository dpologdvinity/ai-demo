"""AdaBoost (Adaptive Boosting) algorithm implementation.

This module implements an AdaBoost classifier using scikit-learn
for multiclass classification tasks with ensemble learning.
"""

import time
from typing import Dict, Any, List
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class AdaBoostModel:
    """AdaBoost classifier wrapper.

    This class provides a clean interface for training and evaluating
    AdaBoost models with configurable hyperparameters. AdaBoost combines
    multiple weak learners (decision trees) by adaptively weighting them
    based on their performance.

    Attributes:
        n_estimators: Number of weak learners to train
        learning_rate: Weight shrinkage applied to each classifier
        algorithm: Boosting algorithm variant ('SAMME' or 'SAMME.R')
        random_state: Random seed for reproducibility
        model: Trained scikit-learn AdaBoostClassifier instance
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        algorithm: str = 'SAMME.R',
        random_state: int = 42
    ):
        """Initialize AdaBoost model with hyperparameters.

        Args:
            n_estimators: Number of weak learners (boosting rounds)
            learning_rate: Weight shrinkage (smaller values need more estimators)
            algorithm: Boosting algorithm ('SAMME' or 'SAMME.R')
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state

        # Use decision stumps (depth=1) as weak learners
        base_estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)

        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train the AdaBoost model and evaluate performance.

        This method trains the ensemble of weak learners sequentially,
        with each learner focusing on samples that previous learners
        misclassified. It also tracks learning progress across boosting rounds.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing:
                - metrics: Performance metrics (accuracy, precision, recall, f1)
                - predictions: Predictions on test set
                - execution_time_ms: Training time in milliseconds
                - feature_importance: Feature importance scores
                - learning_curve: Learning progress across boosting rounds
                - confusion_matrix: Confusion matrix array
        """
        start_time = time.time()

        # Train the model
        self.model.fit(X_train, y_train)

        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)

        # Calculate metrics
        # For multiclass, use weighted average
        average = 'binary' if len(np.unique(y_train)) == 2 else 'weighted'

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average=average, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average=average, zero_division=0)),
            'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(y_test, y_pred))
        }

        # Get feature importance
        feature_importance = self.model.feature_importances_

        # Generate learning curve data
        learning_curve_data = self._generate_learning_curve(X_train, y_train, X_test, y_test)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        return {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'execution_time_ms': execution_time,
            'feature_importance': feature_importance,
            'learning_curve': learning_curve_data,
            'confusion_matrix': cm,
            'classes': self.model.classes_.tolist()
        }

    def _generate_learning_curve(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, List[float]]:
        """Generate learning curve data showing performance vs number of estimators.

        Evaluates model performance at different stages of boosting to show
        how accuracy improves with more weak learners.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing:
                - estimators: Number of estimators at each evaluation point
                - train_scores: Training accuracy at each point
                - test_scores: Test accuracy at each point
                - estimator_errors: Error rate for each estimator
                - estimator_weights: Weight assigned to each estimator
        """
        # Evaluate at 10 evenly spaced points
        n_points = min(10, self.n_estimators)
        estimator_counts = np.linspace(1, self.n_estimators, n_points, dtype=int)

        train_scores = []
        test_scores = []

        for n_estimators in estimator_counts:
            # Use staged predict to get predictions at this stage
            # staged_predict returns a generator, so we need to iterate through it
            stage_train_pred = None
            stage_test_pred = None

            for i, (train_pred, test_pred) in enumerate(
                zip(
                    self.model.staged_predict(X_train),
                    self.model.staged_predict(X_test)
                ),
                start=1
            ):
                if i == n_estimators:
                    stage_train_pred = train_pred
                    stage_test_pred = test_pred
                    break

            if stage_train_pred is not None and stage_test_pred is not None:
                train_score = accuracy_score(y_train, stage_train_pred)
                test_score = accuracy_score(y_test, stage_test_pred)
                train_scores.append(float(train_score))
                test_scores.append(float(test_score))

        # Get estimator errors and weights
        estimator_errors = self.model.estimator_errors_.tolist() if hasattr(self.model, 'estimator_errors_') else []
        estimator_weights = self.model.estimator_weights_.tolist() if hasattr(self.model, 'estimator_weights_') else []

        return {
            'estimators': estimator_counts.tolist(),
            'train_scores': train_scores,
            'test_scores': test_scores,
            'estimator_errors': estimator_errors,
            'estimator_weights': estimator_weights
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

        Only available when using SAMME.R algorithm.

        Args:
            X: Features to predict on

        Returns:
            Predicted class probabilities

        Raises:
            AttributeError: If algorithm is SAMME (doesn't support probabilities)
        """
        if self.algorithm == 'SAMME':
            raise AttributeError("SAMME algorithm doesn't support predict_proba. Use SAMME.R instead.")
        return self.model.predict_proba(X)

    def get_estimator_performance(self) -> Dict[str, Any]:
        """Get detailed performance information for each weak learner.

        Returns:
            Dictionary containing:
                - n_estimators: Total number of estimators trained
                - estimator_errors: Error rate for each estimator
                - estimator_weights: Weight assigned to each estimator
        """
        return {
            'n_estimators': len(self.model.estimators_),
            'estimator_errors': self.model.estimator_errors_.tolist() if hasattr(self.model, 'estimator_errors_') else [],
            'estimator_weights': self.model.estimator_weights_.tolist() if hasattr(self.model, 'estimator_weights_') else []
        }
