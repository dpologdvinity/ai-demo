"""Hierarchical Clustering model implementation using scikit-learn.

This module provides a HierarchicalClusteringModel class that wraps scikit-learn's
AgglomerativeClustering for building a hierarchy of clusters using linkage methods.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import time


class HierarchicalClusteringModel:
    """Hierarchical Clustering implementation for unsupervised learning.

    This class provides training and clustering functionality for Hierarchical Clustering,
    an unsupervised learning algorithm that builds a hierarchy of clusters using linkage
    methods. It can produce both a dendrogram and cluster assignments.

    Attributes:
        model: Scikit-learn AgglomerativeClustering instance
        n_clusters: Number of clusters to find
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        affinity: Distance metric used
        training_time_ms: Time taken to train the model in milliseconds
    """

    def __init__(
        self,
        n_clusters: int = 3,
        linkage: str = 'ward',
        affinity: str = 'euclidean'
    ):
        """Initialize Hierarchical Clustering model.

        Args:
            n_clusters: The number of clusters to find. Must be at least 2.
            linkage: Which linkage criterion to use. Options:
                - 'ward': Minimizes variance within clusters (only with 'euclidean')
                - 'complete': Maximum distances between all observations
                - 'average': Average distances between all observations
                - 'single': Minimum distances between all observations
            affinity: Distance metric used. Options:
                - 'euclidean': L2 distance
                - 'manhattan': L1 distance
                - 'cosine': Cosine similarity

        Raises:
            ValueError: If n_clusters < 2 or invalid linkage/affinity combination
        """
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be at least 2, got {n_clusters}")

        # Ward linkage only works with euclidean distance
        if linkage == 'ward' and affinity != 'euclidean':
            raise ValueError("Ward linkage requires euclidean affinity")

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=affinity
        )
        self.training_time_ms = 0.0
        self.linkage_matrix = None

    def train(self, X: np.ndarray) -> Dict[str, Any]:
        """Train the Hierarchical Clustering model.

        Fits the hierarchical clustering model to the data and computes the
        linkage matrix for dendrogram visualization.

        Args:
            X: Training features, shape (n_samples, n_features)

        Returns:
            Dictionary containing:
                - n_clusters: Number of clusters found
                - n_samples: Number of samples processed
                - training_time_ms: Time taken to train in milliseconds
                - linkage_method: Linkage criterion used
                - affinity: Distance metric used

        Raises:
            ValueError: If X is empty or has fewer samples than clusters
        """
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples ({X.shape[0]}) must be >= n_clusters ({self.n_clusters})"
            )

        start_time = time.time()

        # Fit the model
        self.model.fit(X)

        # Compute linkage matrix for dendrogram
        # We need to recompute using scipy's linkage function
        self.linkage_matrix = linkage(X, method=self.linkage, metric=self.affinity)

        self.training_time_ms = (time.time() - start_time) * 1000

        return {
            "n_clusters": self.n_clusters,
            "n_samples": X.shape[0],
            "training_time_ms": self.training_time_ms,
            "linkage_method": self.linkage,
            "affinity": self.affinity
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get cluster assignments for data.

        Note: Hierarchical clustering is not a predictive model in the traditional
        sense. This method returns the cluster labels from the fitted model.
        For new data, you would need to refit the model.

        Args:
            X: Features to cluster, shape (n_samples, n_features)

        Returns:
            Cluster labels, shape (n_samples,)

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'labels_'):
            raise ValueError("Model must be trained before getting cluster assignments")

        # For hierarchical clustering, we return the fitted labels
        # Note: This assumes X is the same data used for training
        return self.model.labels_

    def get_cluster_labels(self) -> np.ndarray:
        """Get cluster labels from the trained model.

        Returns:
            Array of cluster labels for each sample

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'labels_'):
            raise ValueError("Model must be trained before getting cluster labels")

        return self.model.labels_

    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality using multiple metrics.

        Computes various clustering metrics to assess the quality of the clustering.
        All metrics are computed on the training data since hierarchical clustering
        is not a predictive model.

        Args:
            X: Features that were used for clustering, shape (n_samples, n_features)

        Returns:
            Dictionary containing:
                - silhouette_score: Mean silhouette coefficient (-1 to 1, higher is better)
                - davies_bouldin_score: DB index (lower is better)
                - calinski_harabasz_score: Variance ratio (higher is better)

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'labels_'):
            raise ValueError("Model must be trained before evaluation")

        labels = self.model.labels_

        # Silhouette score: [-1, 1], higher is better
        silhouette = silhouette_score(X, labels)

        # Davies-Bouldin index: [0, inf), lower is better
        davies_bouldin = davies_bouldin_score(X, labels)

        # Calinski-Harabasz index: [0, inf), higher is better
        calinski_harabasz = calinski_harabasz_score(X, labels)

        return {
            "silhouette_score": float(silhouette),
            "davies_bouldin_score": float(davies_bouldin),
            "calinski_harabasz_score": float(calinski_harabasz)
        }

    def get_dendrogram_data(self) -> Dict[str, Any]:
        """Get dendrogram data for visualization.

        Returns:
            Dictionary containing linkage matrix and parameters for dendrogram plotting

        Raises:
            ValueError: If model has not been trained yet
        """
        if self.linkage_matrix is None:
            raise ValueError("Model must be trained before getting dendrogram data")

        return {
            "linkage_matrix": self.linkage_matrix.tolist(),
            "n_clusters": self.n_clusters,
            "linkage_method": self.linkage,
            "affinity": self.affinity
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters.

        Returns:
            Dictionary containing model configuration and learned parameters

        Raises:
            ValueError: If model has not been trained yet
        """
        if not hasattr(self.model, 'labels_'):
            raise ValueError("Model must be trained to get model info")

        # Count samples in each cluster
        unique_labels, counts = np.unique(self.model.labels_, return_counts=True)
        cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist()))

        return {
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "affinity": self.affinity,
            "n_samples": len(self.model.labels_),
            "cluster_sizes": cluster_sizes,
            "n_connected_components": self.model.n_connected_components_,
            "n_features_in": int(self.model.n_features_in_)
        }
