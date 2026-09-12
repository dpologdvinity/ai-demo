"""Pydantic schemas for Elastic Net Regression API endpoints.

This module defines request and response models for the Elastic Net Regression
algorithm endpoints, ensuring type safety and validation.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ElasticNetRequest(BaseModel):
    """Request schema for Elastic Net Regression training.

    Attributes:
        alpha: Overall regularization strength. Higher values create more
            regularization. Must be positive.
        l1_ratio: Mix between L1 and L2 penalties. For l1_ratio = 0 the penalty
            is Ridge (L2). For l1_ratio = 1 it is Lasso (L1). For values in
            between, it combines both penalties.
        max_iter: Maximum number of iterations for the optimization algorithm.
        fit_intercept: Whether to calculate the intercept for this model.
        dataset_name: Name of the dataset to use. Defaults to 'boston'
            (California Housing dataset).
        test_size: Proportion of dataset to use for testing (0.0 to 1.0).
        normalize: Whether to normalize features before regression using
            StandardScaler. Recommended for features with different scales.

    Example:
        >>> request = ElasticNetRequest(
        ...     alpha=1.0,
        ...     l1_ratio=0.5,
        ...     max_iter=1000,
        ...     fit_intercept=True,
        ...     dataset_name="boston",
        ...     test_size=0.3,
        ...     normalize=True
        ... )
    """

    alpha: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Overall regularization strength"
    )
    l1_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Mix between L1 (Lasso) and L2 (Ridge) penalties"
    )
    max_iter: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum number of iterations"
    )
    fit_intercept: bool = Field(
        default=True,
        description="Whether to calculate the intercept"
    )
    dataset_name: str = Field(
        default="boston",
        description="Dataset name to use for training"
    )
    test_size: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="Proportion of dataset to use for testing"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize features before regression"
    )


class ElasticNetResponse(BaseModel):
    """Response schema for Elastic Net Regression training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics (R² score, MSE, RMSE, MAE)
        predictions: Model predictions on test set
        actual: Actual target values from test set
        visualization_data: Data formatted for frontend visualization including:
            - chart_data: Predictions vs actual values
            - coefficient_data: Feature coefficients showing both sparsity and shrinkage
            - regularization_path: Effect of different l1_ratio values
        execution_time_ms: Total execution time in milliseconds
        model_info: Information about the trained model including:
            - coefficients: Feature weights
            - intercept: Bias term
            - n_nonzero_coefs: Number of non-zero coefficients
            - sparsity: Percentage of zero coefficients
            - l2_norm: L2 norm of coefficients
        feature_importance: Dictionary mapping feature names to importance scores
        parameters_used: Parameters that were used for training
        error: Error message if training failed

    Example:
        >>> response = ElasticNetResponse(
        ...     success=True,
        ...     metrics={"r2_score": 0.78, "mse": 0.16, "rmse": 0.40, "mae": 0.31},
        ...     predictions=[1.2, 3.4, 5.6],
        ...     actual=[1.1, 3.3, 5.5],
        ...     visualization_data={...},
        ...     execution_time_ms=58.4,
        ...     model_info={...},
        ...     feature_importance={...},
        ...     parameters_used={...}
        ... )
    """

    success: bool
    metrics: Dict[str, float] = Field(default_factory=dict)
    predictions: Optional[List[float]] = None
    actual: Optional[List[float]] = None
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0.0
    model_info: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "metrics": {
                    "r2_score": 0.7543210987654321,
                    "mse": 0.5123456789012345,
                    "rmse": 0.7157890123456789,
                    "mae": 0.5234567890123456
                },
                "predictions": [2.1, 1.6, 3.3],
                "actual": [2.3, 1.7, 3.0],
                "visualization_data": {
                    "chart_data": [
                        {"index": 0, "predicted": 2.1, "actual": 2.3},
                        {"index": 1, "predicted": 1.6, "actual": 1.7},
                        {"index": 2, "predicted": 3.3, "actual": 3.0}
                    ],
                    "coefficient_data": [
                        {"feature": "MedInc", "coefficient": 0.82, "magnitude": 0.82, "selected": True},
                        {"feature": "HouseAge", "coefficient": 0.11, "magnitude": 0.11, "selected": True},
                        {"feature": "AveRooms", "coefficient": 0.0, "magnitude": 0.0, "selected": False},
                        {"feature": "AveBedrms", "coefficient": -0.31, "magnitude": 0.31, "selected": True}
                    ],
                    "regularization_path": [
                        {"l1_ratio": 0.0, "type": "Ridge", "r2_score": 0.76, "n_nonzero": 8, "sparsity": 0.0},
                        {"l1_ratio": 0.5, "type": "Elastic Net", "r2_score": 0.75, "n_nonzero": 7, "sparsity": 12.5},
                        {"l1_ratio": 1.0, "type": "Lasso", "r2_score": 0.72, "n_nonzero": 6, "sparsity": 25.0}
                    ]
                },
                "execution_time_ms": 58.4,
                "model_info": {
                    "alpha": 1.0,
                    "l1_ratio": 0.5,
                    "coefficients": [0.82, 0.11, 0.0, -0.31],
                    "intercept": 2.07,
                    "n_features_in": 8,
                    "n_nonzero_coefs": 7,
                    "sparsity": 12.5,
                    "l2_norm": 0.89,
                    "converged": True
                },
                "feature_importance": {
                    "MedInc": 0.82,
                    "HouseAge": 0.11,
                    "AveRooms": 0.0,
                    "AveBedrms": 0.31
                },
                "parameters_used": {
                    "alpha": 1.0,
                    "l1_ratio": 0.5,
                    "max_iter": 1000,
                    "fit_intercept": True,
                    "normalize": True
                }
            }
        }
