"""Gaussian Mixture Model (GMM) implementation using scikit-learn.

This module provides a GaussianMixtureModelClass that wraps scikit-learn's
GaussianMixture for probabilistic clustering of data points.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import time


class GaussianMixtureModelClass:
    """Gaussian Mixture Model implementation for probabilistic clustering.

    GMM assumes data is generated from a mixture of Gaussian distributions
    with unknown parameters. It uses the Expectation-Maximization (EM) algorithm
    to find the maximum likelihood estimates of these parameters.

    Attributes:
        model: Scikit-learn GaussianMixture instance
        n_components: Number of Gaussian components (clusters)
        covariance_type: Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
        max_iter: Maximum number of EM iterations
        training_time_ms: Time taken to fit the model in milliseconds
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = 'full',
        max_iter: int = 100,
        random_state: int = 42
    ):
        """Initialize Gaussian Mixture Model.

        Args:
            n_components: Number of mixture components (clusters). Must be >= 2.
            covariance_type: Type of covariance parameters to use.
                Options: 'full' (each component has its own covariance matrix),
                        'tied' (all components share the same covariance matrix),
                        'diag' (diagonal covariance matrices),
                        'spherical' (single variance per component)
            max_iter: Maximum number of EM iterations to perform.
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If n_components < 2 or covariance_type is invalid.
        """
        if n_components < 2:
            raise ValueError(f"n_components must be at least 2, got {n_components}")

        valid_cov_types = ['full', 'tied', 'diag', 'spherical']
        if covariance_type not in valid_cov_types:
            raise ValueError(
                f"covariance_type must be one of {valid_cov_types}, got {covariance_type}"
            )

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state
        )
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.training_time_ms = 0.0

    def fit(self, X: np.ndarray) -> Dict[str, Any]:
        """Fit the Gaussian Mixture Model to data.

        Uses the Expectation-Maximization (EM) algorithm to estimate
        the parameters of the Gaussian mixture components.

        Args:
            X: Training data, shape (n_samples, n_features)

        Returns:
            Dictionary containing:
                - converged: Whether EM algorithm converged
                - n_iter: Number of EM iterations performed
                - lower_bound: Lower bound value on the log-likelihood
                - training_time_ms: Time taken to fit in milliseconds

        Raises:
            ValueError: If X is empty or has wrong dimensionality.
        """
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")

        start_time = time.time()
        self.model.fit(X)
        self.training_time_ms = (time.time() - start_time) * 1000

        return {
            "converged": bool(self.model.converged_),
            "n_iter": int(self.model.n_iter_),
            "lower_bound": float(self.model.lower_bound_),
            "training_time_ms": self.training_time_ms
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for samples.

        Args:
            X: Samples to predict, shape (n_samples, n_features)

        Returns:
            Predicted cluster labels, shape (n_samples,)

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not hasattr(self.model, 'converged_'):
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability of each component for samples.

        Args:
            X: Samples to predict probabilities for, shape (n_samples, n_features)

        Returns:
            Probability of each component for each sample,
            shape (n_samples, n_components)

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not hasattr(self.model, 'converged_'):
            raise ValueError("Model must be fitted before predicting probabilities")

        return self.model.predict_proba(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute log-likelihood of each sample under the model.

        Args:
            X: Samples to score, shape (n_samples, n_features)

        Returns:
            Log-likelihood of each sample, shape (n_samples,)

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not hasattr(self.model, 'converged_'):
            raise ValueError("Model must be fitted before scoring samples")

        return self.model.score_samples(X)

    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality using multiple metrics.

        Computes silhouette score and Davies-Bouldin index to assess
        the quality of the clustering.

        Args:
            X: Samples to evaluate, shape (n_samples, n_features)

        Returns:
            Dictionary containing:
                - silhouette_score: Silhouette coefficient (higher is better, range: -1 to 1)
                - davies_bouldin_score: Davies-Bouldin index (lower is better)
                - bic: Bayesian Information Criterion (lower is better)
                - aic: Akaike Information Criterion (lower is better)
                - log_likelihood: Average log-likelihood per sample

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not hasattr(self.model, 'converged_'):
            raise ValueError("Model must be fitted before evaluation")

        labels = self.predict(X)

        # Calculate silhouette score (requires at least 2 samples per cluster)
        try:
            silhouette = silhouette_score(X, labels)
        except ValueError:
            silhouette = 0.0

        # Calculate Davies-Bouldin index
        try:
            davies_bouldin = davies_bouldin_score(X, labels)
        except ValueError:
            davies_bouldin = float('inf')

        return {
            "silhouette_score": float(silhouette),
            "davies_bouldin_score": float(davies_bouldin),
            "bic": float(self.model.bic(X)),
            "aic": float(self.model.aic(X)),
            "log_likelihood": float(self.model.score(X))
        }

    def get_parameters(self) -> Dict[str, Any]:
        """Get the learned parameters of the Gaussian mixture components.

        Returns:
            Dictionary containing:
                - means: Mean of each mixture component, shape (n_components, n_features)
                - covariances: Covariance of each component (shape depends on covariance_type)
                - weights: Weights of each mixture component, shape (n_components,)
                - precisions_cholesky: Cholesky decomposition of precision matrices

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not hasattr(self.model, 'converged_'):
            raise ValueError("Model must be fitted to get parameters")

        return {
            "means": self.model.means_.tolist(),
            "covariances": self._format_covariances(),
            "weights": self.model.weights_.tolist(),
            "n_components": self.n_components,
            "covariance_type": self.covariance_type
        }

    def _format_covariances(self) -> List[Any]:
        """Format covariance matrices for JSON serialization.

        Returns:
            List representation of covariance matrices appropriate for the
            covariance_type used.
        """
        if self.covariance_type == 'full':
            return self.model.covariances_.tolist()
        elif self.covariance_type == 'tied':
            return self.model.covariances_.tolist()
        elif self.covariance_type == 'diag':
            return self.model.covariances_.tolist()
        elif self.covariance_type == 'spherical':
            return self.model.covariances_.tolist()
        else:
            return []

    def generate_contour_data(
        self,
        X: np.ndarray,
        n_points: int = 100
    ) -> Dict[str, Any]:
        """Generate data for plotting probability contours.

        Creates a meshgrid over the data range and computes probability
        densities for visualization.

        Args:
            X: Original data used for determining plot bounds, shape (n_samples, 2)
            n_points: Number of points per dimension in the meshgrid

        Returns:
            Dictionary containing:
                - x_grid: X coordinates of the meshgrid
                - y_grid: Y coordinates of the meshgrid
                - densities: Probability densities at each grid point

        Raises:
            ValueError: If X doesn't have exactly 2 features.
        """
        if X.shape[1] != 2:
            raise ValueError(f"Contour data can only be generated for 2D data, got {X.shape[1]} features")

        if not hasattr(self.model, 'converged_'):
            raise ValueError("Model must be fitted before generating contour data")

        # Create meshgrid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        x_grid = np.linspace(x_min, x_max, n_points)
        y_grid = np.linspace(y_min, y_max, n_points)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Compute densities
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        log_densities = self.model.score_samples(grid_points)
        densities = np.exp(log_densities).reshape(xx.shape)

        return {
            "x_grid": x_grid.tolist(),
            "y_grid": y_grid.tolist(),
            "densities": densities.tolist()
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.

        Returns:
            Dictionary containing all model configuration and state information.

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not hasattr(self.model, 'converged_'):
            raise ValueError("Model must be fitted to get model info")

        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "max_iter": self.max_iter,
            "converged": bool(self.model.converged_),
            "n_iter": int(self.model.n_iter_),
            "n_features": int(self.model.means_.shape[1])
        }
