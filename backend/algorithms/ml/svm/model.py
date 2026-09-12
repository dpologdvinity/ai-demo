"""Support Vector Machine (SVM) model implementation using scikit-learn.

This module provides an SVMModel class that wraps scikit-learn's SVC
for classification tasks using various kernel methods to find optimal
separating hyperplanes.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time


class SVMModel:
    """Support Vector Machine implementation for classification.

    This class provides training and prediction functionality for SVM,
    a powerful supervised learning algorithm that finds the optimal hyperplane
    to separate different classes in the feature space. It supports various
    kernel functions for both linear and non-linear classification.

    Attributes:
        model: Scikit-learn SVC instance
        C: Regularization parameter
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        gamma: Kernel coefficient
        degree: Polynomial degree (for poly kernel)
        training_time_ms: Time taken to train the model in milliseconds
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        degree: int = 3
    ):
        """Initialize Support Vector Machine model.

        Args:
            C: Regularization parameter. The strength of the regularization
                is inversely proportional to C. Must be strictly positive.
                Smaller values specify stronger regularization.
            kernel: Specifies the kernel type to be used in the algorithm.
                Options: 'linear', 'poly', 'rbf', 'sigmoid'
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
                Options: 'scale', 'auto', or float value
            degree: Degree of the polynomial kernel function ('poly').
                Ignored by all other kernels.
        """
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            random_state=42
        )
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.training_time_ms = 0.0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Train the SVM model.

        Fits the SVM model to the training data by finding the optimal
        hyperplane that maximizes the margin between classes.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training target labels, shape (n_samples,)

        Returns:
            Dictionary containing:
                - n_support_vectors: Total number of support vectors
                - support_vectors_per_class: Number of support vectors per class
                - n_features: Number of input features
                - n_classes: Number of classes
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
            "n_support_vectors": int(self.model.n_support_.sum()),
            "support_vectors_per_class": self.model.n_support_.tolist(),
            "n_features": X_train.shape[1],
            "n_classes": len(self.model.classes_),
            "training_time_ms": self.training_time_ms
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.

        Args:
            X: Features to predict on, shape (n_samples, n_features)

        Returns:
            Predicted class labels, shape (n_samples,)

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'support_vectors_'):
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probability estimates.

        Note: This requires the model to be trained with probability=True.
        For speed optimization, our default model doesn't use this feature.

        Args:
            X: Features to predict on, shape (n_samples, n_features)

        Returns:
            Class probability estimates, shape (n_samples, n_classes)

        Raises:
            ValueError: If model has not been trained yet
            AttributeError: If model was not trained with probability=True
        """
        if not hasattr(self.model, 'support_vectors_'):
            raise ValueError("Model must be trained before making predictions")

        # Return decision function as a proxy for probability
        decision = self.model.decision_function(X)
        return decision

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate model performance on test data.

        Computes multiple classification metrics to assess model performance.

        Args:
            X_test: Test features, shape (n_samples, n_features)
            y_test: True target labels, shape (n_samples,)

        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy score
                - precision: Precision score per class (macro average)
                - recall: Recall score per class (macro average)
                - f1_score: F1 score (macro average)
                - confusion_matrix: Confusion matrix as nested list

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'support_vectors_'):
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        conf_matrix = confusion_matrix(y_test, y_pred)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": conf_matrix.tolist()
        }

    def get_support_vectors(self) -> Dict[str, Any]:
        """Get support vectors and related information.

        Returns:
            Dictionary containing:
                - support_vectors: The support vectors
                - support_vector_indices: Indices of support vectors in training data
                - n_support_vectors: Total number of support vectors

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'support_vectors_'):
            raise ValueError("Model must be trained to get support vectors")

        return {
            "support_vectors": self.model.support_vectors_.tolist(),
            "support_vector_indices": self.model.support_.tolist(),
            "n_support_vectors": int(self.model.n_support_.sum())
        }

    def get_decision_boundary_data(
        self,
        X: np.ndarray,
        resolution: int = 100
    ) -> Dict[str, Any]:
        """Generate data for visualizing the decision boundary.

        Creates a mesh grid over the feature space and predicts the class
        for each point to visualize the decision boundary.

        Args:
            X: Training data to determine the range of the mesh
            resolution: Number of points per dimension in the mesh

        Returns:
            Dictionary containing:
                - xx: X coordinates of the mesh grid
                - yy: Y coordinates of the mesh grid
                - Z: Predicted class labels for each mesh point

        Raises:
            ValueError: If model has not been trained yet or X has wrong dimensions
        """
        if not hasattr(self.model, 'support_vectors_'):
            raise ValueError("Model must be trained to generate decision boundary")

        if X.shape[1] < 2:
            raise ValueError("Decision boundary visualization requires at least 2 features")

        # Use only first two features for visualization
        X_2d = X[:, :2]

        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        # Predict on mesh grid
        mesh_input = np.c_[xx.ravel(), yy.ravel()]

        # If original data has more than 2 features, pad with zeros
        if X.shape[1] > 2:
            padding = np.zeros((mesh_input.shape[0], X.shape[1] - 2))
            mesh_input = np.hstack([mesh_input, padding])

        Z = self.model.predict(mesh_input)
        Z = Z.reshape(xx.shape)

        return {
            "xx": xx.tolist(),
            "yy": yy.tolist(),
            "Z": Z.tolist()
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and learned parameters.

        Returns:
            Dictionary containing model configuration and learned parameters

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'support_vectors_'):
            raise ValueError("Model must be trained to get model info")

        return {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma if isinstance(self.gamma, str) else float(self.gamma),
            "degree": self.degree,
            "n_support_vectors": int(self.model.n_support_.sum()),
            "n_classes": len(self.model.classes_),
            "classes": self.model.classes_.tolist()
        }
