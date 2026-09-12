"""
K-Nearest Neighbors (KNN) algorithm implementation.

This module implements the KNN classification algorithm using scikit-learn,
including training, prediction, evaluation, and visualization data generation.
"""

import time
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from .data import load_and_prepare_data, reduce_to_2d
from .schema import KNNRequest, KNNResponse, KNNMetrics, VisualizationData


class KNNModel:
    """K-Nearest Neighbors classification model.

    This class encapsulates the KNN algorithm, providing methods for training,
    prediction, and generating visualization data. KNN is an instance-based
    learning algorithm that classifies new points based on the majority vote
    of their K nearest neighbors in the feature space.

    Attributes:
        model: The scikit-learn KNeighborsClassifier instance
        params: Dictionary of model parameters
        X_train: Training features (scaled)
        X_test: Test features (scaled)
        y_train: Training labels
        y_test: Test labels
        target_names: Names of classification classes
        feature_names: Names of input features
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        metric: str = 'euclidean',
        p: int = 2
    ):
        """Initialize KNN model with specified parameters.

        Args:
            n_neighbors: Number of neighbors to consider (K value)
            weights: Weight function ('uniform' or 'distance')
            metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
            p: Power parameter for Minkowski metric
        """
        self.params = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'metric': metric,
            'p': p if metric == 'minkowski' else 2
        }

        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=self.params['p']
        )

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_names = None
        self.feature_names = None

    def train(
        self,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the KNN model on Iris dataset.

        Loads the Iris dataset, trains the KNN classifier, makes predictions,
        and computes evaluation metrics.

        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - metrics: Performance metrics (accuracy, precision, recall, f1)
                - predictions: Predicted labels for test set
                - train_accuracy: Accuracy on training set
                - test_accuracy: Accuracy on test set

        Raises:
            RuntimeError: If training fails
        """
        # Load and prepare data
        data = load_and_prepare_data(test_size=test_size, random_state=random_state)
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        self.target_names = data['target_names']
        self.feature_names = data['feature_names']

        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        precision = precision_score(self.y_test, y_pred_test, average='weighted')
        recall = recall_score(self.y_test, y_pred_test, average='weighted')
        f1 = f1_score(self.y_test, y_pred_test, average='weighted')

        return {
            'metrics': {
                'accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            },
            'predictions': y_pred_test.tolist(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for new data points.

        Args:
            X: Feature array of shape (n_samples, n_features)

        Returns:
            Array of predicted class labels

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions")

        return self.model.predict(X)

    def generate_visualization_data(self) -> Dict[str, Any]:
        """Generate data for frontend visualization.

        Creates 2D scatter plot data using PCA dimensionality reduction,
        confusion matrix, and decision boundary information.

        Returns:
            Dictionary containing:
                - scatter_data: List of points with x, y, class, and prediction
                - confusion_matrix: 2D confusion matrix
                - class_labels: Names of classes
                - decision_boundary: Optional boundary data for 2D plots

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("Model must be trained before generating visualizations")

        # Reduce to 2D for visualization using PCA
        X_test_2d = reduce_to_2d(self.X_test, method='pca')
        y_pred = self.model.predict(self.X_test)

        # Create scatter plot data
        scatter_data = []
        for i in range(len(X_test_2d)):
            scatter_data.append({
                'x': float(X_test_2d[i, 0]),
                'y': float(X_test_2d[i, 1]),
                'class': int(self.y_test[i]),
                'predicted': int(y_pred[i]),
                'label': self.target_names[self.y_test[i]]
            })

        # Generate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Generate decision boundary for 2D visualization
        decision_boundary = self._generate_decision_boundary(X_test_2d, y_pred)

        return {
            'scatter_data': scatter_data,
            'confusion_matrix': cm.tolist(),
            'class_labels': self.target_names,
            'decision_boundary': decision_boundary
        }

    def _generate_decision_boundary(
        self,
        X_2d: np.ndarray,
        y_pred: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate decision boundary visualization for 2D data.

        Creates a mesh grid over the 2D feature space and predicts the class
        for each point to visualize the decision boundaries.

        Args:
            X_2d: 2D projected features
            y_pred: Predicted labels

        Returns:
            List of dictionaries with x, y, and predicted class for grid points
        """
        # Train a new KNN on 2D data for boundary visualization
        X_train_2d = reduce_to_2d(self.X_train, method='pca')
        boundary_model = KNeighborsClassifier(**self.params)
        boundary_model.fit(X_train_2d, self.y_train)

        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        h = 0.1  # step size in the mesh

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        # Predict class for each point in mesh
        Z = boundary_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Sample the grid for frontend (send subset to avoid large payload)
        step = 5  # Only send every 5th point
        boundary_data = []
        for i in range(0, xx.shape[0], step):
            for j in range(0, xx.shape[1], step):
                boundary_data.append({
                    'x': float(xx[i, j]),
                    'y': float(yy[i, j]),
                    'class': int(Z[i, j])
                })

        return boundary_data


def train_knn(request: KNNRequest) -> KNNResponse:
    """Train KNN model with given parameters and return results.

    This is the main entry point for training a KNN model. It handles the
    complete workflow: initialization, training, prediction, and visualization
    data generation.

    Args:
        request: KNNRequest containing algorithm parameters

    Returns:
        KNNResponse with training results, metrics, and visualization data

    Raises:
        Exception: If training fails for any reason
    """
    start_time = time.time()

    try:
        # Initialize model
        model = KNNModel(
            n_neighbors=request.n_neighbors,
            weights=request.weights,
            metric=request.metric,
            p=request.p
        )

        # Train model
        results = model.train(
            test_size=request.test_size,
            random_state=request.random_state
        )

        # Generate visualization data
        viz_data = model.generate_visualization_data()

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Build response
        response = KNNResponse(
            success=True,
            metrics=KNNMetrics(**results['metrics']),
            predictions=results['predictions'],
            actual=model.y_test.tolist(),
            visualization_data=VisualizationData(**viz_data),
            execution_time_ms=execution_time_ms,
            parameters={
                'n_neighbors': request.n_neighbors,
                'weights': request.weights,
                'metric': request.metric,
                'p': request.p,
                'test_size': request.test_size
            },
            message=f"KNN trained successfully with K={request.n_neighbors}"
        )

        return response

    except Exception as e:
        raise Exception(f"KNN training failed: {str(e)}")
