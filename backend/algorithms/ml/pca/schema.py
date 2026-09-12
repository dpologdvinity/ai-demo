"""Pydantic schemas for PCA API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PCARequest(BaseModel):
    """Request schema for PCA dimensionality reduction.

    Attributes:
        n_components: Number of principal components to compute (default: 2, range: 2-10)
        whiten: Whether to whiten the components (divide by singular values to ensure
            uncorrelated outputs with unit variance). Default: False
        random_state: Random seed for reproducibility (default: 42)
    """

    n_components: int = Field(
        default=2,
        ge=2,
        le=10,
        description="Number of principal components to compute"
    )
    whiten: bool = Field(
        default=False,
        description="Whether to whiten the components for unit variance"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "n_components": 2,
                "whiten": False,
                "random_state": 42
            }
        }


class PCAResponse(BaseModel):
    """Response schema for PCA dimensionality reduction results.

    Attributes:
        success: Whether the operation completed successfully
        transformed_data: Data points in principal component space (2D or 3D)
        labels: Original labels for colored visualization
        explained_variance: Variance explained by each component
        explained_variance_ratio: Proportion of total variance explained by each component
        cumulative_variance_ratio: Cumulative variance explained up to each component
        visualization_data: Data formatted for frontend visualization including
            scatter plot data and variance plot data
        execution_time_ms: Total execution time in milliseconds
        model_info: Information about the fitted model
        parameters_used: Parameters that were used for the analysis
        error: Error message if operation failed
    """

    success: bool
    transformed_data: Optional[List[List[float]]] = None
    labels: Optional[List[int]] = None
    explained_variance: Optional[List[float]] = None
    explained_variance_ratio: Optional[List[float]] = None
    cumulative_variance_ratio: Optional[List[float]] = None
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0.0
    model_info: Optional[Dict[str, Any]] = None
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "transformed_data": [[1.2, 0.5], [2.3, -0.8], [-1.1, 1.2]],
                "labels": [0, 1, 2],
                "explained_variance": [3.45, 1.23],
                "explained_variance_ratio": [0.45, 0.16],
                "cumulative_variance_ratio": [0.45, 0.61],
                "visualization_data": {
                    "scatter_data": [
                        {"pc1": 1.2, "pc2": 0.5, "label": 0},
                        {"pc1": 2.3, "pc2": -0.8, "label": 1}
                    ],
                    "variance_data": [
                        {"component": "PC1", "variance": 0.45},
                        {"component": "PC2", "variance": 0.16}
                    ]
                },
                "execution_time_ms": 15.3,
                "model_info": {
                    "n_components": 2,
                    "whiten": False,
                    "n_features": 64,
                    "n_samples": 1257
                },
                "parameters_used": {
                    "n_components": 2,
                    "whiten": False
                }
            }
        }
