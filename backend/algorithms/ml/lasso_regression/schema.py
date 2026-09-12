"""Pydantic schemas for Lasso Regression API endpoints.

This module defines request and response models for the Lasso Regression
algorithm endpoints, ensuring type safety and validation.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class LassoRegressionRequest(BaseModel):
    """Request schema for Lasso Regression training.

    Attributes:
        alpha: Regularization strength (L1 penalty). Higher values create
            sparser models with more features eliminated. Must be positive.
        max_iter: Maximum number of iterations for the optimization algorithm.
        selection: Strategy for updating coefficients ('cyclic' or 'random').
            'cyclic' updates sequentially, 'random' updates in random order.
        dataset_name: Name of the dataset to use. Defaults to 'boston'
            (California Housing dataset).
        test_size: Proportion of dataset to use for testing (0.0 to 1.0).
        normalize: Whether to normalize features before regression using
            StandardScaler. Recommended for features with different scales.

    Example:
        >>> request = LassoRegressionRequest(
        ...     alpha=1.0,
        ...     max_iter=1000,
        ...     selection="cyclic",
        ...     dataset_name="boston",
        ...     test_size=0.3,
        ...     normalize=True
        ... )
    """

    alpha: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Regularization strength (L1 penalty)"
    )
    max_iter: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum number of iterations"
    )
    selection: str = Field(
        default="cyclic",
        description="Coefficient update rule: 'cyclic' or 'random'"
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


class LassoRegressionResponse(BaseModel):
    """Response schema for Lasso Regression training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics (R² score, MSE, RMSE, MAE)
        predictions: Model predictions on test set
        actual: Actual target values from test set
        visualization_data: Data formatted for frontend visualization including:
            - chart_data: Predictions vs actual values
            - coefficient_data: Feature coefficients showing sparsity
        execution_time_ms: Total execution time in milliseconds
        model_info: Information about the trained model including:
            - coefficients: Feature weights
            - intercept: Bias term
            - n_nonzero_coefs: Number of selected features
            - sparsity: Percentage of zero coefficients
        feature_importance: Dictionary mapping feature names to importance scores
        parameters_used: Parameters that were used for training
        error: Error message if training failed

    Example:
        >>> response = LassoRegressionResponse(
        ...     success=True,
        ...     metrics={"r2_score": 0.75, "mse": 0.18, "rmse": 0.42, "mae": 0.33},
        ...     predictions=[1.2, 3.4, 5.6],
        ...     actual=[1.1, 3.3, 5.5],
        ...     visualization_data={...},
        ...     execution_time_ms=52.3,
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
                    "r2_score": 0.7143821074982375,
                    "mse": 0.5971635408428947,
                    "rmse": 0.7727554621958177,
                    "mae": 0.5436928419302817
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
                        {"feature": "MedInc", "coefficient": 0.85, "selected": True},
                        {"feature": "HouseAge", "coefficient": 0.12, "selected": True},
                        {"feature": "AveRooms", "coefficient": 0.0, "selected": False},
                        {"feature": "AveBedrms", "coefficient": -0.34, "selected": True}
                    ]
                },
                "execution_time_ms": 52.3,
                "model_info": {
                    "alpha": 1.0,
                    "coefficients": [0.85, 0.12, 0.0, -0.34],
                    "intercept": 2.07,
                    "n_features_in": 8,
                    "n_nonzero_coefs": 6,
                    "sparsity": 25.0,
                    "converged": True
                },
                "feature_importance": {
                    "MedInc": 0.85,
                    "HouseAge": 0.12,
                    "AveRooms": 0.0,
                    "AveBedrms": 0.34
                },
                "parameters_used": {
                    "alpha": 1.0,
                    "max_iter": 1000,
                    "selection": "cyclic",
                    "normalize": True
                }
            }
        }
