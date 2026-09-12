"""Pydantic schemas for Regularization Techniques API endpoints."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RegularizationRequest(BaseModel):
    """Request schema for Regularization Techniques training.

    Attributes:
        technique: Regularization technique to use
        alpha: Regularization strength (lambda parameter)
        l1_ratio: Elastic Net mixing parameter (0.0 = Ridge, 1.0 = Lasso)
        max_iterations: Maximum training iterations
        early_stopping_rounds: Patience for early stopping
        n_samples: Number of samples in synthetic dataset
        n_features: Number of features in synthetic dataset
        normalize: Whether to normalize features before training
    """

    technique: str = Field(
        default='l2',
        description="Regularization technique: 'l1', 'l2', 'elastic_net', 'early_stopping', 'compare'"
    )
    alpha: float = Field(
        default=1.0,
        ge=0.001,
        le=100.0,
        description="Regularization strength (higher = more regularization)"
    )
    l1_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Elastic Net mixing parameter (0.0 = Ridge, 1.0 = Lasso)"
    )
    max_iterations: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum number of training iterations"
    )
    early_stopping_rounds: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Number of rounds with no improvement before stopping"
    )
    n_samples: int = Field(
        default=200,
        ge=100,
        le=500,
        description="Number of samples in synthetic dataset"
    )
    n_features: int = Field(
        default=100,
        ge=50,
        le=200,
        description="Number of features in synthetic dataset"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize features before training"
    )


class RegularizationResponse(BaseModel):
    """Response schema for Regularization Techniques training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics for each technique
        predictions: Model predictions on test set
        actual: Actual target values from test set
        visualization_data: Data for frontend visualizations including:
            - coefficient_paths: How coefficients change with alpha
            - loss_curves: Training and validation loss over iterations
            - sparsity_comparison: Coefficient sparsity across techniques
            - overfitting_curves: Train vs test performance gap
        execution_time_ms: Total execution time in milliseconds
        model_info: Information about trained models
        parameters_used: Parameters that were used for training
        error: Error message if training failed
    """

    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    predictions: Optional[List[float]] = None
    actual: Optional[List[float]] = None
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
                "metrics": {
                    "l1": {"r2_score": 0.72, "mse": 0.45, "sparsity": 65.0},
                    "l2": {"r2_score": 0.75, "mse": 0.42, "sparsity": 0.0},
                    "elastic_net": {"r2_score": 0.74, "mse": 0.43, "sparsity": 45.0},
                    "early_stopping": {"r2_score": 0.76, "mse": 0.41, "optimal_iterations": 87}
                },
                "visualization_data": {
                    "coefficient_paths": [],
                    "loss_curves": [],
                    "sparsity_comparison": [],
                    "overfitting_curves": []
                },
                "execution_time_ms": 523.4,
                "parameters_used": {
                    "technique": "compare",
                    "alpha": 1.0,
                    "l1_ratio": 0.5
                }
            }
        }
