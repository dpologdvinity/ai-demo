"""
Decision Tree Classifier implementation using scikit-learn.

This module implements a Decision Tree classifier for multi-class classification
with comprehensive tree structure extraction for visualization.
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from utils.datasets import DatasetManager


class DecisionTreeModel:
    """Decision Tree Classifier for classification tasks.

    This class implements a Decision Tree classifier using scikit-learn's
    DecisionTreeClassifier. It provides methods for training the model,
    making predictions, and extracting comprehensive metrics and tree structure
    for visualization.

    Attributes:
        model: The trained DecisionTreeClassifier instance
        feature_names: List of feature names from the dataset
        target_names: List of target class names
        training_time_ms: Time taken to train the model in milliseconds
    """

    def __init__(self):
        """Initialize the Decision Tree model."""
        self.model: Optional[DecisionTreeClassifier] = None
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        self.training_time_ms: float = 0.0

    def train(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        dataset_name: str = 'iris',
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the Decision Tree classifier.

        Args:
            max_depth: Maximum depth of the tree. None means unlimited depth.
            min_samples_split: Minimum samples required to split an internal node.
            min_samples_leaf: Minimum samples required to be at a leaf node.
            criterion: Function to measure split quality ('gini' or 'entropy').
            dataset_name: Name of dataset to use (default: 'iris').
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary containing:
                - success: Whether training was successful
                - metrics: Performance metrics on test set
                - predictions: Model predictions on test set
                - visualization_data: Tree structure and confusion matrix data
                - execution_time_ms: Training time in milliseconds
                - parameters_used: Actual parameters used for training

        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If training fails
        """
        try:
            # Validate parameters
            if min_samples_split < 2:
                raise ValueError("min_samples_split must be at least 2")
            if min_samples_leaf < 1:
                raise ValueError("min_samples_leaf must be at least 1")
            if criterion not in ['gini', 'entropy']:
                raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion}")

            # Load dataset
            data = DatasetManager.get_iris(random_state=random_state)
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']
            self.feature_names = data['feature_names']
            self.target_names = data['target_names']

            # Create and train model
            start_time = time.time()

            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=random_state
            )

            self.model.fit(X_train, y_train)

            end_time = time.time()
            self.training_time_ms = (end_time - start_time) * 1000

            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Extract tree structure for visualization
            tree_structure = self._extract_tree_structure()
            tree_text = export_text(
                self.model,
                feature_names=self.feature_names,
                max_depth=3  # Limit text depth for readability
            )

            # Prepare results
            return {
                'success': True,
                'metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'n_nodes': int(self.model.tree_.node_count),
                    'n_leaves': int(self._count_leaves()),
                    'max_depth_achieved': int(self.model.get_depth())
                },
                'predictions': {
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_pred_proba': y_pred_proba.tolist()
                },
                'visualization_data': {
                    'confusion_matrix': cm.tolist(),
                    'labels': self.target_names,
                    'tree_structure': tree_structure,
                    'tree_text': tree_text,
                    'feature_importance': {
                        'features': self.feature_names,
                        'importance': self.model.feature_importances_.tolist()
                    }
                },
                'execution_time_ms': self.training_time_ms,
                'parameters_used': {
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'criterion': criterion,
                    'dataset_name': dataset_name,
                    'random_state': random_state
                }
            }

        except Exception as e:
            return {
                'success': False,
                'metrics': {},
                'predictions': None,
                'visualization_data': {},
                'execution_time_ms': 0.0,
                'parameters_used': {},
                'error': str(e)
            }

    def _extract_tree_structure(self) -> Dict[str, Any]:
        """Extract the complete tree structure for visualization.

        Returns:
            Dictionary representing the tree structure with nodes and edges.
        """
        if self.model is None:
            return {}

        tree = self.model.tree_

        def build_tree_dict(node_id: int = 0) -> Dict[str, Any]:
            """Recursively build tree structure."""
            # Check if leaf node
            if tree.feature[node_id] == -2:  # Leaf node
                return {
                    'id': int(node_id),
                    'type': 'leaf',
                    'class': int(np.argmax(tree.value[node_id][0])),
                    'class_name': self.target_names[int(np.argmax(tree.value[node_id][0]))],
                    'samples': int(tree.n_node_samples[node_id]),
                    'value': tree.value[node_id][0].tolist(),
                    'impurity': float(tree.impurity[node_id])
                }

            # Internal node
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            return {
                'id': int(node_id),
                'type': 'internal',
                'feature': int(tree.feature[node_id]),
                'feature_name': self.feature_names[tree.feature[node_id]],
                'threshold': float(tree.threshold[node_id]),
                'samples': int(tree.n_node_samples[node_id]),
                'impurity': float(tree.impurity[node_id]),
                'left': build_tree_dict(left_child),
                'right': build_tree_dict(right_child)
            }

        return build_tree_dict(0)

    def _count_leaves(self) -> int:
        """Count the number of leaf nodes in the tree.

        Returns:
            Number of leaf nodes.
        """
        if self.model is None:
            return 0

        tree = self.model.tree_
        return int(np.sum(tree.children_left == -1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Feature array to predict.

        Returns:
            Array of predicted class labels.

        Raises:
            RuntimeError: If model hasn't been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probability predictions.

        Args:
            X: Feature array to predict.

        Returns:
            Array of class probabilities.

        Raises:
            RuntimeError: If model hasn't been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions")

        return self.model.predict_proba(X)
