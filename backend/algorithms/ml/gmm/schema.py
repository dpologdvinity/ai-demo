"""Pydantic schemas for Gaussian Mixture Model API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GMMRequest(BaseModel):
    """Request schema for Gaussian Mixture Model training.

    Attributes:
        n_components: Number of Gaussian components (clusters) to fit.
            Default: 3, Range: 2-10
        covariance_type: Type of covariance parameters to use.
            Options: 'full' (each component has its own covariance matrix),
                    'tied' (all components share the same covariance matrix),
                    'diag' (diagonal covariance matrices),
                    'spherical' (single variance per component)
            Default: 'full'
        max_iter: Maximum number of EM algorithm iterations.
            Default: 100, Range: 10-500
        n_samples: Number of samples to generate for blob dataset.
            Default: 300, Range: 100-1000
        random_state: Random seed for reproducibility. Default: 42
    """

    n_components: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of Gaussian components (clusters)"
    )
    covariance_type: str = Field(
        default='full',
        description="Type of covariance: 'full', 'tied', 'diag', or 'spherical'"
    )
    max_iter: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of EM iterations"
    )
    n_samples: int = Field(
        default=300,
        ge=100,
        le=1000,
        description="Number of samples in the generated dataset"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "n_components": 3,
                "covariance_type": "full",
                "max_iter": 100,
                "n_samples": 300,
                "random_state": 42
            }
        }


class GMMResponse(BaseModel):
    """Response schema for Gaussian Mixture Model training results.

    Attributes:
        metrics: Dictionary of clustering quality metrics
            - silhouette_score: Silhouette coefficient (-1 to 1, higher is better)
            - davies_bouldin_score: Davies-Bouldin index (lower is better)
            - bic: Bayesian Information Criterion (lower is better)
            - aic: Akaike Information Criterion (lower is better)
            - log_likelihood: Average log-likelihood per sample
        predictions: Predicted cluster labels for each sample
        probabilities: Probability of each cluster for each sample
        visualization_data: Data for plotting probability contours and clusters
            - x: X coordinates of data points
            - y: Y coordinates of data points
            - labels: Cluster assignments
            - contour: Contour plot data (x_grid, y_grid, densities)
            - means: Cluster centers (mean positions)
        parameters: Learned GMM parameters
            - means: Mean of each Gaussian component
            - covariances: Covariance matrices
            - weights: Mixture weights
        execution_time_ms: Total training time in milliseconds
        model_info: Additional model information
            - n_components: Number of components used
            - covariance_type: Covariance type used
            - converged: Whether EM algorithm converged
            - n_iter: Number of iterations performed
    """

    metrics: Dict[str, float] = Field(
        description="Clustering quality metrics"
    )
    predictions: List[int] = Field(
        description="Predicted cluster labels for each sample"
    )
    probabilities: List[List[float]] = Field(
        description="Probability of each cluster for each sample"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Data for visualizing clusters and probability contours"
    )
    parameters: Dict[str, Any] = Field(
        description="Learned Gaussian Mixture Model parameters"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    model_info: Dict[str, Any] = Field(
        description="Model configuration and convergence information"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metrics": {
                    "silhouette_score": 0.65,
                    "davies_bouldin_score": 0.82,
                    "bic": -1245.32,
                    "aic": -1198.45,
                    "log_likelihood": 4.15
                },
                "predictions": [0, 0, 1, 2, 1, 2],
                "probabilities": [
                    [0.95, 0.03, 0.02],
                    [0.92, 0.05, 0.03],
                    [0.02, 0.94, 0.04]
                ],
                "visualization_data": {
                    "x": [1.2, 1.5, 2.3],
                    "y": [0.5, 0.8, 1.2],
                    "labels": [0, 0, 1],
                    "contour": {
                        "x_grid": [0.0, 0.1, 0.2],
                        "y_grid": [0.0, 0.1, 0.2],
                        "densities": [[0.1, 0.2], [0.15, 0.25]]
                    },
                    "means": [[1.3, 0.6], [2.1, 1.5], [3.2, 2.1]]
                },
                "parameters": {
                    "means": [[1.3, 0.6], [2.1, 1.5], [3.2, 2.1]],
                    "covariances": [
                        [[0.5, 0.1], [0.1, 0.4]],
                        [[0.6, 0.05], [0.05, 0.55]]
                    ],
                    "weights": [0.35, 0.33, 0.32],
                    "n_components": 3,
                    "covariance_type": "full"
                },
                "execution_time_ms": 23.5,
                "model_info": {
                    "n_components": 3,
                    "covariance_type": "full",
                    "max_iter": 100,
                    "converged": True,
                    "n_iter": 42,
                    "n_features": 2
                }
            }
        }
