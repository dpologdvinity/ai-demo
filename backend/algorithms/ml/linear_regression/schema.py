"""Pydantic schemas for Linear Regression API endpoints.

This module defines request and response models for the Linear Regression
algorithm endpoints, ensuring type safety and validation.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class LinearRegressionRequest(BaseModel):
    """Request schema for Linear Regression training.

    Attributes:
        fit_intercept: Whether to calculate the intercept for the model.
            If False, no intercept will be used (data is expected to be centered).
        normalize: Whether to normalize features before regression using
            StandardScaler. Recommended for features with different scales.
        dataset_name: Name of the dataset to use. Defaults to 'boston'
            (California Housing dataset).
        test_size: Proportion of dataset to use for testing (0.0 to 1.0).

    Example:
        >>> request = LinearRegressionRequest(
        ...     fit_intercept=True,
        ...     normalize=True,
        ...     dataset_name="boston",
        ...     test_size=0.3
        ... )
    """

    fit_intercept: bool = Field(
        default=True,
        description="Whether to calculate the intercept for this model"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize features before regression"
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


class LinearRegressionResponse(BaseModel):
    """Response schema for Linear Regression training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics (R² score, MSE, RMSE, MAE)
        predictions: Model predictions on test set
        actual: Actual target values from test set
        visualization_data: Data formatted for frontend visualization
            containing indices, predictions, and actual values
        execution_time_ms: Total execution time in milliseconds
        model_info: Information about the trained model including
            coefficients and intercept
        parameters_used: Parameters that were used for training
        error: Error message if training failed

    Example:
        >>> response = LinearRegressionResponse(
        ...     success=True,
        ...     metrics={"r2_score": 0.85, "mse": 0.15, "rmse": 0.39, "mae": 0.31},
        ...     predictions=[1.2, 3.4, 5.6],
        ...     actual=[1.1, 3.3, 5.5],
        ...     visualization_data={...},
        ...     execution_time_ms=45.3,
        ...     model_info={...},
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
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "metrics": {
                    "r2_score": 0.7406426641094095,
                    "mse": 0.5558915986952444,
                    "rmse": 0.7455861943508663,
                    "mae": 0.5308926804618569
                },
                "predictions": [2.3, 1.8, 3.1],
                "actual": [2.5, 1.7, 3.0],
                "visualization_data": {
                    "chart_data": [
                        {"index": 0, "predicted": 2.3, "actual": 2.5},
                        {"index": 1, "predicted": 1.8, "actual": 1.7},
                        {"index": 2, "predicted": 3.1, "actual": 3.0}
                    ]
                },
                "execution_time_ms": 45.3,
                "model_info": {
                    "coefficients": [0.5, -0.3, 0.8],
                    "intercept": 1.2,
                    "n_features_in": 3
                },
                "parameters_used": {
                    "fit_intercept": True,
                    "normalize": True
                }
            }
        }
