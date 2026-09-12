"""Ensemble Methods implementation.

This module implements various ensemble learning techniques including:
- Bagging: Bootstrap aggregating using BaggingClassifier
- Boosting: Gradient Boosting for sequential error correction
- Stacking: Multi-layer ensemble with meta-learner
- Voting: Hard/soft voting ensemble
"""

import time
from typing import Dict, Any, List, Literal
import numpy as np
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from scipy.stats import pearsonr


class EnsembleMethodsModel:
    """Ensemble Methods classifier wrapper.

    This class provides implementations of multiple ensemble techniques
    and allows comparison between them. It supports Bagging, Boosting,
    Stacking, and Voting ensembles with various base estimators.

    Attributes:
        method: Ensemble method to use
        n_estimators: Number of base models
        base_model_type: Type of base estimator
        max_samples: Sample ratio for bagging
        learning_rate: Learning rate for boosting
        random_state: Random seed for reproducibility
        model: Trained ensemble model
        base_models: Individual base models for comparison
    """

    def __init__(
        self,
        method: Literal['bagging', 'boosting', 'stacking', 'voting', 'all'] = 'voting',
        n_estimators: int = 10,
        base_model: Literal['decision_tree', 'svm', 'knn', 'logistic'] = 'decision_tree',
        max_samples: float = 0.8,
        learning_rate: float = 1.0,
        random_state: int = 42
    ):
        """Initialize Ensemble Methods model.

        Args:
            method: Ensemble method to use
            n_estimators: Number of base models in ensemble
            base_model: Type of base estimator
            max_samples: Sample ratio for bagging
            learning_rate: Learning rate for boosting
            random_state: Random seed
        """
        self.method = method
        self.n_estimators = n_estimators
        self.base_model_type = base_model
        self.max_samples = max_samples
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.base_models = []
        self.base_estimator = self._create_base_estimator()

    def _create_base_estimator(self):
        """Create base estimator based on specified type.

        Returns:
            Scikit-learn estimator instance
        """
        if self.base_model_type == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=5,
                random_state=self.random_state
            )
        elif self.base_model_type == 'svm':
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            )
        elif self.base_model_type == 'knn':
            return KNeighborsClassifier(n_neighbors=5)
        elif self.base_model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown base model type: {self.base_model_type}")

    def _create_ensemble(self, method: str):
        """Create ensemble model based on specified method.

        Args:
            method: Ensemble method name

        Returns:
            Scikit-learn ensemble estimator
        """
        if method == 'bagging':
            return BaggingClassifier(
                estimator=self._create_base_estimator(),
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                random_state=self.random_state
            )
        elif method == 'boosting':
            # Gradient Boosting only works with decision trees
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=3,
                random_state=self.random_state
            )
        elif method == 'voting':
            # Create diverse base models for voting
            estimators = [
                ('dt', DecisionTreeClassifier(max_depth=5, random_state=self.random_state)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=self.random_state)),
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('lr', LogisticRegression(max_iter=1000, random_state=self.random_state))
            ]
            return VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
        elif method == 'stacking':
            # Create base estimators for stacking
            estimators = [
                ('dt', DecisionTreeClassifier(max_depth=5, random_state=self.random_state)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=self.random_state)),
                ('knn', KNeighborsClassifier(n_neighbors=5))
            ]
            # Use logistic regression as meta-learner
            final_estimator = LogisticRegression(max_iter=1000, random_state=self.random_state)
            return StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=5
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train the ensemble model and evaluate performance.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing metrics, predictions, and visualization data
        """
        start_time = time.time()

        # Train the ensemble model
        self.model = self._create_ensemble(self.method)
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)

        # Calculate metrics
        average = 'binary' if len(np.unique(y_train)) == 2 else 'weighted'

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average=average, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average=average, zero_division=0)),
            'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(y_test, y_pred))
        }

        # Train individual base models for comparison
        single_model_results = self._train_single_models(X_train, y_train, X_test, y_test)

        # Get feature importance
        feature_importance = self._get_feature_importance()

        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(X_test, y_test)

        # Get voting patterns
        voting_data = self._get_voting_patterns(X_test, y_test, y_pred)

        # Get individual predictions
        individual_predictions = self._get_individual_predictions(X_test)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        execution_time = (time.time() - start_time) * 1000

        return {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'execution_time_ms': execution_time,
            'feature_importance': feature_importance,
            'confusion_matrix': cm,
            'single_model_results': single_model_results,
            'diversity_metrics': diversity_metrics,
            'voting_data': voting_data,
            'individual_predictions': individual_predictions,
            'classes': np.unique(y_train).tolist()
        }

    def _train_single_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Train individual base models for performance comparison.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            List of dictionaries containing metrics for each model
        """
        results = []
        average = 'binary' if len(np.unique(y_train)) == 2 else 'weighted'

        # Define models to compare
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
            'SVM': SVC(kernel='rbf', random_state=self.random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.base_models.append((name, model))

            results.append({
                'model_name': name,
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average=average, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average=average, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average=average, zero_division=0))
            })

        return results

    def _get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the ensemble.

        Returns:
            Feature importance array
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            elif self.method == 'voting' or self.method == 'stacking':
                # For voting/stacking, average importance from tree-based models
                importances = []
                if self.method == 'voting':
                    for name, estimator in self.model.named_estimators_.items():
                        if hasattr(estimator, 'feature_importances_'):
                            importances.append(estimator.feature_importances_)
                elif self.method == 'stacking':
                    for name, estimator in self.model.named_estimators_.items():
                        if hasattr(estimator, 'feature_importances_'):
                            importances.append(estimator.feature_importances_)

                if importances:
                    return np.mean(importances, axis=0)

            # If no feature importance available, return zeros
            return np.zeros(self.model.n_features_in_)
        except:
            # Fallback: return zeros if error occurs
            try:
                return np.zeros(self.model.n_features_in_)
            except:
                return np.zeros(10)  # Default size

    def _calculate_diversity_metrics(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Calculate diversity metrics between base models.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing diversity metrics
        """
        if not self.base_models:
            return {
                'disagreement': 0.0,
                'avg_correlation': 0.0,
                'q_statistic': 0.0
            }

        # Get predictions from all base models
        predictions = []
        for _, model in self.base_models:
            pred = model.predict(X_test)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate disagreement (proportion of samples where models disagree)
        n_samples = predictions.shape[1]
        disagreement_count = 0
        for i in range(n_samples):
            sample_preds = predictions[:, i]
            if len(np.unique(sample_preds)) > 1:
                disagreement_count += 1
        disagreement = disagreement_count / n_samples

        # Calculate average correlation between predictions
        correlations = []
        n_models = len(predictions)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                try:
                    corr, _ = pearsonr(predictions[i], predictions[j])
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    pass
        avg_correlation = float(np.mean(correlations)) if correlations else 0.0

        # Calculate Q-statistic (pairwise diversity measure)
        q_statistics = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Calculate agreement patterns
                n11 = np.sum((predictions[i] == y_test) & (predictions[j] == y_test))
                n00 = np.sum((predictions[i] != y_test) & (predictions[j] != y_test))
                n10 = np.sum((predictions[i] == y_test) & (predictions[j] != y_test))
                n01 = np.sum((predictions[i] != y_test) & (predictions[j] == y_test))

                denominator = (n11 * n00 + n01 * n10)
                if denominator > 0:
                    q = (n11 * n00 - n01 * n10) / denominator
                    q_statistics.append(q)

        q_statistic = float(np.mean(q_statistics)) if q_statistics else 0.0

        return {
            'disagreement': float(disagreement),
            'avg_correlation': avg_correlation,
            'q_statistic': q_statistic
        }

    def _get_voting_patterns(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ensemble_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Get voting patterns and agreement data.

        Args:
            X_test: Test features
            y_test: Test labels
            ensemble_pred: Ensemble predictions

        Returns:
            Dictionary containing voting pattern data
        """
        if not self.base_models:
            return {
                'sample_indices': [],
                'model_predictions': [],
                'ensemble_predictions': [],
                'actual_labels': [],
                'agreement_scores': []
            }

        # Get predictions from all models
        model_predictions = []
        for _, model in self.base_models:
            pred = model.predict(X_test)
            model_predictions.append(pred.tolist())

        # Calculate agreement scores (proportion of models agreeing with ensemble)
        model_predictions_array = np.array(model_predictions)
        agreement_scores = []
        for i in range(len(ensemble_pred)):
            sample_preds = model_predictions_array[:, i]
            agreement = np.sum(sample_preds == ensemble_pred[i]) / len(self.base_models)
            agreement_scores.append(float(agreement))

        # Sample 50 points for visualization
        n_samples = min(50, len(y_test))
        indices = np.random.choice(len(y_test), n_samples, replace=False)

        return {
            'sample_indices': indices.tolist(),
            'model_predictions': [[int(pred[i]) for pred in model_predictions] for i in indices],
            'ensemble_predictions': [int(ensemble_pred[i]) for i in indices],
            'actual_labels': [int(y_test[i]) for i in indices],
            'agreement_scores': [agreement_scores[i] for i in indices]
        }

    def _get_individual_predictions(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model.

        Args:
            X_test: Test features

        Returns:
            Dictionary mapping model names to their predictions
        """
        predictions = {}
        for name, model in self.base_models:
            pred = model.predict(X_test)
            predictions[name] = pred

        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Predicted class labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Features to predict on

        Returns:
            Predicted class probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"Model {self.method} does not support predict_proba")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.

        Returns:
            Dictionary containing model information
        """
        info = {
            'method': self.method,
            'n_estimators': self.n_estimators,
            'base_model': self.base_model_type,
            'n_base_models': len(self.base_models)
        }

        if self.method == 'bagging':
            info['max_samples'] = self.max_samples
        elif self.method == 'boosting':
            info['learning_rate'] = self.learning_rate

        return info
