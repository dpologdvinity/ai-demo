"""t-SNE model implementation using scikit-learn.

This module provides a TSNEModel class that wraps scikit-learn's TSNE for
non-linear dimensionality reduction, particularly useful for visualizing
high-dimensional data in 2D or 3D space.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import time


class TSNEModel:
    """t-SNE implementation for non-linear dimensionality reduction.

    t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear
    dimensionality reduction technique that is particularly well-suited for
    embedding high-dimensional data into a low-dimensional space of two or
    three dimensions for visualization. It models pairwise similarities between
    data points in high-dimensional space and finds a low-dimensional embedding
    that preserves these similarities.

    Attributes:
        model: Scikit-learn TSNE instance
        scaler: StandardScaler for data normalization
        n_components: Number of dimensions in embedded space (2 or 3)
        perplexity: Balance between local and global aspects of data
        learning_rate: Learning rate for optimization
        n_iter: Number of optimization iterations
        training_time_ms: Time taken to fit the model in milliseconds
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: int = 42
    ):
        """Initialize t-SNE model.

        Args:
            n_components: Dimension of the embedded space (2 or 3).
            perplexity: Related to the number of nearest neighbors considered.
                Larger datasets usually require a larger perplexity. Typical
                values are between 5 and 50.
            learning_rate: Learning rate for the optimization. Typical values
                are between 10 and 1000.
            n_iter: Maximum number of iterations for optimization. Should be
                at least 250.
            random_state: Random seed for reproducibility.
        """
        self.model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state,
            init='pca',  # Initialize with PCA for better results
            method='barnes_hut'  # Faster approximation for large datasets
        )
        self.scaler = StandardScaler()
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.training_time_ms = 0.0

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> Dict[str, Any]:
        """Fit t-SNE model and transform the data to embedded space.

        Computes the t-SNE embedding by minimizing the Kullback-Leibler
        divergence between the joint probabilities in high-dimensional space
        and the low-dimensional embedding.

        Args:
            X: Input features, shape (n_samples, n_features)
            y: Labels for visualization (optional), shape (n_samples,)

        Returns:
            Dictionary containing:
                - embedded_data: Data in t-SNE embedded space
                - labels: Original labels (if provided)
                - n_components: Number of embedding dimensions
                - training_time_ms: Time taken to fit in milliseconds
                - kl_divergence: Final Kullback-Leibler divergence

        Raises:
            ValueError: If X is empty, has invalid shape, or perplexity
                is too large for the number of samples
        """
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples, got {X.shape[0]}")
        if self.perplexity >= X.shape[0]:
            raise ValueError(
                f"Perplexity ({self.perplexity}) must be less than "
                f"number of samples ({X.shape[0]})"
            )

        # Normalize the data (important for t-SNE)
        X_scaled = self.scaler.fit_transform(X)

        # Fit and transform
        start_time = time.time()
        X_embedded = self.model.fit_transform(X_scaled)
        self.training_time_ms = (time.time() - start_time) * 1000

        result = {
            "embedded_data": X_embedded,
            "labels": y.tolist() if y is not None else None,
            "n_components": self.n_components,
            "training_time_ms": self.training_time_ms,
            "kl_divergence": self.model.kl_divergence_
        }

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration parameters.

        Returns:
            Dictionary containing model configuration and learned parameters

        Raises:
            ValueError: If model has not been fitted yet
        """
        if not hasattr(self.model, 'kl_divergence_'):
            raise ValueError("Model must be fitted to get model info")

        return {
            "n_components": self.n_components,
            "perplexity": self.perplexity,
            "learning_rate": self.learning_rate,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
            "n_iter_final": self.model.n_iter_,
            "kl_divergence": float(self.model.kl_divergence_)
        }

    def prepare_visualization_data(
        self,
        embedded_data: np.ndarray,
        labels: np.ndarray = None
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualization.

        Args:
            embedded_data: t-SNE embedded data (n_samples, 2 or 3)
            labels: Class labels for coloring points

        Returns:
            Dictionary with formatted data for scatter plot visualization
        """
        if self.n_components == 2:
            # 2D scatter plot data
            scatter_data = []
            for i in range(len(embedded_data)):
                point = {
                    "x": float(embedded_data[i, 0]),
                    "y": float(embedded_data[i, 1]),
                    "label": int(labels[i]) if labels is not None else 0
                }
                scatter_data.append(point)

            return {
                "type": "2d",
                "scatter_data": scatter_data,
                "x_label": "t-SNE Component 1",
                "y_label": "t-SNE Component 2"
            }

        elif self.n_components == 3:
            # 3D scatter plot data
            scatter_data = []
            for i in range(len(embedded_data)):
                point = {
                    "x": float(embedded_data[i, 0]),
                    "y": float(embedded_data[i, 1]),
                    "z": float(embedded_data[i, 2]),
                    "label": int(labels[i]) if labels is not None else 0
                }
                scatter_data.append(point)

            return {
                "type": "3d",
                "scatter_data": scatter_data,
                "x_label": "t-SNE Component 1",
                "y_label": "t-SNE Component 2",
                "z_label": "t-SNE Component 3"
            }

        else:
            raise ValueError(f"Unsupported n_components: {self.n_components}")
