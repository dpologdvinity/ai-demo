"""DBSCAN Clustering algorithm implementation."""

import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from .schema import DBSCANParameters, DBSCANResponse, ClusterInfo
from .data import get_dbscan_data, prepare_visualization_data


class DBSCANModel:
    """DBSCAN Clustering algorithm implementation.

    This class implements the DBSCAN (Density-Based Spatial Clustering of
    Applications with Noise) algorithm using scikit-learn, providing methods
    for training and evaluating the model on synthetic data.

    DBSCAN is a density-based unsupervised learning algorithm that groups
    together points that are closely packed together (points with many nearby
    neighbors), marking points in low-density regions as outliers. Unlike
    K-Means, DBSCAN can discover clusters of arbitrary shape and automatically
    determines the number of clusters.

    Key advantages:
    - Discovers arbitrarily shaped clusters
    - Identifies noise/outliers
    - Doesn't require specifying number of clusters in advance
    - Robust to outliers

    Attributes:
        model: The underlying scikit-learn DBSCAN model
        parameters: Training parameters used
        labels: Cluster assignments for training data (-1 for noise)
        n_clusters: Number of clusters found (excluding noise)
        n_noise: Number of noise points detected

    Example:
        >>> params = DBSCANParameters(eps=0.5, min_samples=5)
        >>> dbscan = DBSCANModel()
        >>> response = dbscan.train(params)
        >>> print(f"Found {response.metrics['n_clusters']} clusters")
    """

    def __init__(self):
        """Initialize the DBSCAN model."""
        self.model: Optional[DBSCAN] = None
        self.parameters: Optional[DBSCANParameters] = None
        self.labels: Optional[np.ndarray] = None
        self.n_clusters: int = 0
        self.n_noise: int = 0

    def train(self, parameters: DBSCANParameters) -> DBSCANResponse:
        """Train DBSCAN clustering model.

        Generates synthetic moons data and applies DBSCAN clustering algorithm
        to discover density-based clusters. Computes performance metrics and
        prepares visualization data.

        The moons dataset is specifically chosen to demonstrate DBSCAN's
        strength in finding non-convex clusters that traditional algorithms
        like K-Means would fail to separate properly.

        Args:
            parameters: Training parameters including eps, min_samples, metric

        Returns:
            DBSCANResponse containing training results, metrics, and visualization data

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If training fails

        Example:
            >>> params = DBSCANParameters(eps=0.3, min_samples=10, n_samples=500)
            >>> model = DBSCANModel()
            >>> result = model.train(params)
            >>> if result.success:
            ...     print(f"Training completed in {result.execution_time_ms:.2f}ms")
            ...     print(f"Found {result.metrics['n_clusters']} clusters")
        """
        try:
            start_time = time.time()
            self.parameters = parameters

            # Generate synthetic moons data
            data = get_dbscan_data(
                n_samples=parameters.n_samples,
                noise=parameters.noise,
                random_state=parameters.random_state
            )
            X = data['X']

            # Initialize and train DBSCAN model
            self.model = DBSCAN(
                eps=parameters.eps,
                min_samples=parameters.min_samples,
                metric=parameters.metric
            )

            # Fit the model (DBSCAN doesn't have separate fit/predict)
            self.labels = self.model.fit_predict(X)

            # Count clusters and noise points
            # DBSCAN labels noise points as -1
            unique_labels = set(self.labels)
            self.n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            self.n_noise = int(np.sum(self.labels == -1))

            # Calculate metrics
            metrics = self._calculate_metrics(X)

            # Prepare cluster information
            clusters = self._prepare_cluster_info()

            # Prepare visualization data
            visualization_data = prepare_visualization_data(X, self.labels)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            return DBSCANResponse(
                success=True,
                metrics=metrics,
                clusters=clusters,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump()
            )

        except ValueError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return DBSCANResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                error=f"Invalid parameters: {str(e)}"
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return DBSCANResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                error=f"Training failed: {str(e)}"
            )

    def _calculate_metrics(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate clustering performance metrics.

        Computes various metrics to evaluate DBSCAN clustering quality:
        - Number of clusters found
        - Number of noise points detected
        - Silhouette score (if applicable)

        Args:
            X: Feature array

        Returns:
            Dictionary of metric names to values

        Note:
            Silhouette score is only calculated if there are at least 2 clusters
            and not all points are noise. Score ranges from -1 to 1, where
            higher values indicate better-defined clusters.
        """
        metrics = {
            'n_clusters': self.n_clusters,
            'n_noise': self.n_noise,
            'noise_ratio': float(self.n_noise) / len(X) if len(X) > 0 else 0.0
        }

        # Calculate silhouette score (requires at least 2 clusters)
        # and cannot be computed when all points are noise
        if self.n_clusters >= 2:
            # Filter out noise points for silhouette calculation
            non_noise_mask = self.labels != -1
            if np.sum(non_noise_mask) > 0:
                try:
                    X_non_noise = X[non_noise_mask]
                    labels_non_noise = self.labels[non_noise_mask]

                    # Silhouette score requires at least 2 samples per cluster
                    unique_labels, counts = np.unique(labels_non_noise, return_counts=True)
                    if len(unique_labels) >= 2 and all(counts >= 2):
                        silhouette_avg = silhouette_score(X_non_noise, labels_non_noise)
                        metrics['silhouette_score'] = float(silhouette_avg)
                except Exception:
                    # Silhouette score may fail in edge cases
                    pass

        return metrics

    def _prepare_cluster_info(self) -> list[ClusterInfo]:
        """Prepare detailed information about each cluster.

        Collects statistics about each cluster including the number of points
        assigned to it. Includes noise points as a special cluster with ID -1.

        Returns:
            List of ClusterInfo objects, one per cluster plus noise
        """
        cluster_info = []
        unique_labels = set(self.labels)

        for label in sorted(unique_labels):
            cluster_size = int(np.sum(self.labels == label))
            is_noise = (label == -1)

            cluster_info.append(
                ClusterInfo(
                    cluster_id=int(label),
                    size=cluster_size,
                    is_noise=is_noise
                )
            )

        return cluster_info

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data.

        Note: DBSCAN is not naturally designed for prediction on new data
        as it's based on the density of the original training data.
        This method is provided for interface consistency but may not
        be meaningful for all use cases.

        Args:
            X: Feature array for new data points

        Returns:
            Array of cluster assignments (will be -1 for points not fitting
            any existing cluster density)

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # DBSCAN doesn't have a predict method, so we return the training labels
        # In practice, you might use a KNN approach to assign new points
        # to existing clusters or mark them as noise
        raise NotImplementedError(
            "DBSCAN does not support prediction on new data. "
            "Consider using HDBSCAN for incremental clustering."
        )

    def get_cluster_labels(self) -> Optional[np.ndarray]:
        """Get the cluster labels from training.

        Returns:
            Array of cluster assignments, or None if not trained.
            Noise points are labeled as -1.
        """
        return self.labels
