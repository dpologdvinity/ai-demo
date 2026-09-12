"""Random Forest algorithm implementation using scikit-learn."""

import time
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from .data import load_wine_data
from .schema import RandomForestRequest, RandomForestResponse


class RandomForestModel:
    """Random Forest classifier for multi-class classification.

    Random Forest is an ensemble learning method that constructs multiple decision
    trees during training and outputs the class that is the mode of the classes
    predicted by individual trees. This approach helps reduce overfitting and
    improves generalization compared to a single decision tree.

    Key Features:
        - Ensemble of decision trees
        - Bootstrap aggregating (bagging)
        - Random feature selection at each split
        - Robust to overfitting
        - Provides feature importance scores

    Attributes:
        model: Trained RandomForestClassifier instance
        feature_names: List of feature names from the dataset
        target_names: List of target class names
    """

    def __init__(self):
        """Initialize the Random Forest model."""
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        self.target_names: List[str] = []

    def train(self, request: RandomForestRequest) -> RandomForestResponse:
        """Train Random Forest model and return predictions with metrics.

        Args:
            request: RandomForestRequest containing hyperparameters

        Returns:
            RandomForestResponse containing metrics, predictions, feature importance,
            confusion matrix, and execution time

        Raises:
            ValueError: If invalid parameters are provided
        """
        start_time = time.time()

        # Load wine dataset
        data = load_wine_data()
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        self.feature_names = data['feature_names']
        self.target_names = data['target_names']

        # Convert max_features parameter
        max_features = self._convert_max_features(request.max_features)

        # Initialize and train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            max_features=max_features,
            random_state=request.random_state,
            n_jobs=-1  # Use all available cores
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)

        # Get feature importance
        feature_importance = self._get_feature_importance()

        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Prepare model info
        model_info = {
            'n_estimators': request.n_estimators,
            'max_depth': request.max_depth if request.max_depth else 'None',
            'min_samples_split': request.min_samples_split,
            'max_features': request.max_features,
            'total_trees': len(self.model.estimators_),
            'n_features': len(self.feature_names),
            'n_classes': len(self.target_names),
            'class_names': self.target_names
        }

        return RandomForestResponse(
            metrics=metrics,
            predictions=y_pred.tolist(),
            feature_importance=feature_importance,
            confusion_matrix=cm,
            execution_time_ms=execution_time_ms,
            model_info=model_info
        )

    def _convert_max_features(self, max_features_str: str) -> Optional[str]:
        """Convert max_features string parameter to appropriate type.

        Args:
            max_features_str: String representation ('sqrt', 'log2', 'None')

        Returns:
            Appropriate value for sklearn RandomForestClassifier
        """
        if max_features_str.lower() == 'none':
            return None
        return max_features_str

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary containing accuracy, precision, recall, and f1_score
        """
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            return {}

        importances = self.model.feature_importances_
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, importances)
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Feature array for prediction

        Returns:
            Predicted class labels

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.

        Returns:
            Dictionary containing model configuration and metadata

        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        return {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'max_features': self.model.max_features,
            'n_features': len(self.feature_names),
            'n_classes': len(self.target_names),
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
