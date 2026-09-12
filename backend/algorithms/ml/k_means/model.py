"""K-Means Clustering algorithm implementation."""

import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .schema import KMeansParameters, KMeansResponse, ClusterInfo
from .data import get_kmeans_data, prepare_visualization_data


class KMeansModel:
    """K-Means Clustering algorithm implementation.

    This class implements the K-Means clustering algorithm using scikit-learn,
    providing methods for training and evaluating the model on synthetic data.

    K-Means is an unsupervised learning algorithm that groups data points into
    K clusters by iteratively assigning points to the nearest centroid and
    updating centroid positions.

    Attributes:
        model: The underlying scikit-learn KMeans model
        parameters: Training parameters used
        labels: Cluster assignments for training data
        centers: Coordinates of cluster centroids
        inertia: Sum of squared distances to nearest cluster center

    Example:
        >>> params = KMeansParameters(n_clusters=3, max_iter=300)
        >>> kmeans = KMeansModel()
        >>> response = kmeans.train(params)
        >>> print(f"Inertia: {response.metrics['inertia']:.2f}")
    """

    def __init__(self):
        """Initialize the K-Means model."""
        self.model: Optional[KMeans] = None
        self.parameters: Optional[KMeansParameters] = None
        self.labels: Optional[np.ndarray] = None
        self.centers: Optional[np.ndarray] = None
        self.inertia: Optional[float] = None

    def train(self, parameters: KMeansParameters) -> KMeansResponse:
        """Train K-Means clustering model.

        Generates synthetic blob data and applies K-Means clustering algorithm
        to discover clusters. Computes performance metrics and prepares
        visualization data.

        Args:
            parameters: Training parameters including n_clusters, max_iter, etc.

        Returns:
            KMeansResponse containing training results, metrics, and visualization data

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If training fails

        Example:
            >>> params = KMeansParameters(n_clusters=4, n_samples=500)
            >>> model = KMeansModel()
            >>> result = model.train(params)
            >>> if result.success:
            ...     print(f"Training completed in {result.execution_time_ms:.2f}ms")
        """
        try:
            start_time = time.time()
            self.parameters = parameters

            # Generate synthetic data
            data = get_kmeans_data(
                n_samples=parameters.n_samples,
                n_clusters=parameters.n_clusters,
                random_state=parameters.random_state
            )
            X = data['X']

            # Initialize and train K-Means model
            self.model = KMeans(
                n_clusters=parameters.n_clusters,
                max_iter=parameters.max_iter,
                n_init=parameters.n_init,
                random_state=parameters.random_state
            )

            # Fit the model
            self.model.fit(X)

            # Get clustering results
            self.labels = self.model.labels_
            self.centers = self.model.cluster_centers_
            self.inertia = self.model.inertia_
            n_iterations = self.model.n_iter_

            # Calculate metrics
            metrics = self._calculate_metrics(X)

            # Prepare cluster information
            clusters = self._prepare_cluster_info()

            # Prepare visualization data
            visualization_data = prepare_visualization_data(
                X, self.labels, self.centers
            )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            return KMeansResponse(
                success=True,
                metrics=metrics,
                clusters=clusters,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                n_iterations=int(n_iterations)
            )

        except ValueError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return KMeansResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                error=f"Invalid parameters: {str(e)}"
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return KMeansResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                error=f"Training failed: {str(e)}"
            )

    def _calculate_metrics(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate clustering performance metrics.

        Computes inertia (within-cluster sum of squares) and silhouette score
        (measure of cluster separation and cohesion).

        Args:
            X: Feature array

        Returns:
            Dictionary of metric names to values

        Note:
            Silhouette score ranges from -1 to 1, where higher values indicate
            better-defined clusters.
        """
        metrics = {
            'inertia': float(self.inertia),
        }

        # Calculate silhouette score (requires at least 2 clusters)
        if self.parameters.n_clusters >= 2 and len(X) > self.parameters.n_clusters:
            try:
                silhouette_avg = silhouette_score(X, self.labels)
                metrics['silhouette_score'] = float(silhouette_avg)
            except Exception:
                # Silhouette score may fail if clusters are too small
                metrics['silhouette_score'] = 0.0

        return metrics

    def _prepare_cluster_info(self) -> list[ClusterInfo]:
        """Prepare detailed information about each cluster.

        Collects statistics about each cluster including centroid coordinates
        and the number of points assigned to it.

        Returns:
            List of ClusterInfo objects, one per cluster
        """
        cluster_info = []

        for i in range(self.parameters.n_clusters):
            cluster_size = int(np.sum(self.labels == i))
            center = self.centers[i].tolist()

            cluster_info.append(
                ClusterInfo(
                    cluster_id=i,
                    center=center,
                    size=cluster_size
                )
            )

        return cluster_info

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data.

        Args:
            X: Feature array for new data points

        Returns:
            Array of cluster assignments

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        return self.model.predict(X)

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get the coordinates of cluster centers.

        Returns:
            Array of cluster center coordinates, or None if not trained
        """
        return self.centers
