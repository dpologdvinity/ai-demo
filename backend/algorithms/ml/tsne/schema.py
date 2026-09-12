"""Pydantic schemas for t-SNE API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TSNERequest(BaseModel):
    """Request schema for t-SNE dimensionality reduction.

    Attributes:
        n_components: Number of dimensions for embedding (default: 2, options: [2, 3])
        perplexity: Balance between local and global aspects (default: 30, range: 5-50)
        learning_rate: Learning rate for gradient descent (default: 200, range: 10-1000)
        n_iter: Number of optimization iterations (default: 1000, range: 250-5000)
        random_state: Random seed for reproducibility (default: 42)
    """

    n_components: int = Field(
        default=2,
        ge=2,
        le=3,
        description="Number of dimensions for the embedding (2 or 3)"
    )
    perplexity: float = Field(
        default=30.0,
        ge=5.0,
        le=50.0,
        description="Balance between local and global data structure"
    )
    learning_rate: float = Field(
        default=200.0,
        ge=10.0,
        le=1000.0,
        description="Learning rate for the optimization algorithm"
    )
    n_iter: int = Field(
        default=1000,
        ge=250,
        le=5000,
        description="Maximum number of iterations for optimization"
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
                "perplexity": 30.0,
                "learning_rate": 200.0,
                "n_iter": 1000,
                "random_state": 42
            }
        }


class TSNEResponse(BaseModel):
    """Response schema for t-SNE dimensionality reduction results.

    Attributes:
        success: Whether the operation completed successfully
        embedded_data: Data points in t-SNE embedded space (2D or 3D)
        labels: Original labels for colored visualization
        kl_divergence: Final Kullback-Leibler divergence value
        visualization_data: Data formatted for frontend visualization including
            scatter plot data with appropriate labels
        execution_time_ms: Total execution time in milliseconds
        model_info: Information about the fitted model
        parameters_used: Parameters that were used for the embedding
        error: Error message if operation failed
    """

    success: bool
    embedded_data: Optional[List[List[float]]] = None
    labels: Optional[List[int]] = None
    kl_divergence: Optional[float] = None
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
                "embedded_data": [[1.2, 0.5], [2.3, -0.8], [-1.1, 1.2]],
                "labels": [0, 1, 2, 0, 1, 2],
                "kl_divergence": 1.234,
                "visualization_data": {
                    "type": "2d",
                    "scatter_data": [
                        {"x": 1.2, "y": 0.5, "label": 0},
                        {"x": 2.3, "y": -0.8, "label": 1}
                    ],
                    "x_label": "t-SNE Component 1",
                    "y_label": "t-SNE Component 2"
                },
                "execution_time_ms": 850.3,
                "model_info": {
                    "n_components": 2,
                    "perplexity": 30.0,
                    "learning_rate": 200.0,
                    "n_iter": 1000,
                    "n_iter_final": 1000,
                    "kl_divergence": 1.234
                },
                "parameters_used": {
                    "n_components": 2,
                    "perplexity": 30.0,
                    "learning_rate": 200.0,
                    "n_iter": 1000
                }
            }
        }
