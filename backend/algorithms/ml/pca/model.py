"""Principal Component Analysis model implementation using scikit-learn.

This module provides a PCAModel class that wraps scikit-learn's PCA for
dimensionality reduction, finding the principal components that capture
the most variance in the data.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time


class PCAModel:
    """Principal Component Analysis implementation for dimensionality reduction.

    This class provides functionality for reducing the dimensionality of data
    by finding principal components - orthogonal directions of maximum variance.
    PCA is commonly used for visualization, noise reduction, and feature extraction.

    Attributes:
        model: Scikit-learn PCA instance
        scaler: StandardScaler for data normalization
        n_components: Number of components to keep
        whiten: Whether to whiten the components
        training_time_ms: Time taken to fit the model in milliseconds
    """

    def __init__(self, n_components: int = 2, whiten: bool = False):
        """Initialize PCA model.

        Args:
            n_components: Number of components to keep. Must be between 1 and
                number of features in the data.
            whiten: When True, the components are divided by n_samples times
                components to ensure uncorrelated outputs with unit variance.
        """
        self.model = PCA(n_components=n_components, whiten=whiten)
        self.scaler = StandardScaler()
        self.n_components = n_components
        self.whiten = whiten
        self.training_time_ms = 0.0

    def fit_transform(
        self,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """Fit PCA model and transform the data.

        Fits the PCA model to the data by computing the principal components
        and transforms the data to the new coordinate system.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Dictionary containing:
                - transformed_data: Data in principal component space
                - explained_variance: Variance explained by each component
                - explained_variance_ratio: Proportion of variance explained
                - cumulative_variance_ratio: Cumulative variance explained
                - n_components: Number of components used
                - training_time_ms: Time taken to fit in milliseconds

        Raises:
            ValueError: If X is empty or has invalid shape
        """
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[0] < self.n_components:
            raise ValueError(
                f"n_components ({self.n_components}) must be <= n_samples ({X.shape[0]})"
            )
        if X.shape[1] < self.n_components:
            raise ValueError(
                f"n_components ({self.n_components}) must be <= n_features ({X.shape[1]})"
            )

        # Normalize the data
        X_scaled = self.scaler.fit_transform(X)

        # Fit and transform
        start_time = time.time()
        X_transformed = self.model.fit_transform(X_scaled)
        self.training_time_ms = (time.time() - start_time) * 1000

        # Calculate cumulative variance ratio
        cumulative_variance = np.cumsum(self.model.explained_variance_ratio_)

        return {
            "transformed_data": X_transformed,
            "explained_variance": self.model.explained_variance_.tolist(),
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": cumulative_variance.tolist(),
            "n_components": self.n_components,
            "training_time_ms": self.training_time_ms
        }

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using the fitted PCA model.

        Args:
            X: Features to transform, shape (n_samples, n_features)

        Returns:
            Transformed data in principal component space

        Raises:
            ValueError: If model has not been fitted yet
        """
        if not hasattr(self.model, 'components_'):
            raise ValueError("Model must be fitted before transforming data")

        X_scaled = self.scaler.transform(X)
        return self.model.transform(X_scaled)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Transform data back to original space.

        This is useful for visualization and understanding what information
        is retained by the principal components.

        Args:
            X_transformed: Data in principal component space

        Returns:
            Data transformed back to original feature space

        Raises:
            ValueError: If model has not been fitted yet
        """
        if not hasattr(self.model, 'components_'):
            raise ValueError("Model must be fitted before inverse transforming")

        X_scaled = self.model.inverse_transform(X_transformed)
        return self.scaler.inverse_transform(X_scaled)

    def get_components(self) -> np.ndarray:
        """Get the principal components (eigenvectors).

        Returns:
            Principal components, shape (n_components, n_features)

        Raises:
            ValueError: If model has not been fitted yet
        """
        if not hasattr(self.model, 'components_'):
            raise ValueError("Model must be fitted to get components")

        return self.model.components_

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and learned parameters.

        Returns:
            Dictionary containing model configuration and learned parameters

        Raises:
            ValueError: If model has not been fitted yet
        """
        if not hasattr(self.model, 'components_'):
            raise ValueError("Model must be fitted to get model info")

        return {
            "n_components": self.n_components,
            "whiten": self.whiten,
            "n_features": self.model.n_features_in_,
            "n_samples": self.model.n_samples_,
            "explained_variance": self.model.explained_variance_.tolist(),
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "singular_values": self.model.singular_values_.tolist(),
        }
