"""Pydantic schemas for Ridge Regression API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RidgeRegressionRequest(BaseModel):
    """Request schema for Ridge Regression training.

    Attributes:
        alpha: Regularization strength (default: 1.0, range: 0.01-100.0)
        fit_intercept: Whether to calculate the intercept (default: True)
        solver: Solver to use in computational routines
            Options: 'auto', 'svd', 'cholesky', 'lsqr' (default: 'auto')
    """

    alpha: float = Field(
        default=1.0,
        ge=0.01,
        le=100.0,
        description="Regularization strength. Larger values specify stronger regularization."
    )
    fit_intercept: bool = Field(
        default=True,
        description="Whether to calculate the intercept for this model"
    )
    solver: str = Field(
        default='auto',
        description="Solver to use: 'auto', 'svd', 'cholesky', or 'lsqr'"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "alpha": 1.0,
                "fit_intercept": True,
                "solver": "auto"
            }
        }


class RidgeRegressionResponse(BaseModel):
    """Response schema for Ridge Regression training results.

    Attributes:
        metrics: Dictionary of performance metrics (r2_score, mse, rmse, mae)
        predictions: List of predicted values for test set
        actual_values: List of actual values from test set
        coefficients: Dictionary mapping feature names to coefficient values
        coefficient_analysis: Analysis of coefficient magnitudes
        visualization_data: Data formatted for frontend visualization
        execution_time_ms: Training and prediction time in milliseconds
        model_info: Additional model information (alpha, intercept, etc.)
    """

    metrics: Dict[str, float] = Field(
        description="Performance metrics including r2_score, mse, rmse, mae"
    )
    predictions: List[float] = Field(
        description="Predicted values for the test set"
    )
    actual_values: List[float] = Field(
        description="Actual values from the test set"
    )
    coefficients: Dict[str, float] = Field(
        description="Coefficient values for each feature"
    )
    coefficient_analysis: Dict[str, Any] = Field(
        description="Analysis of coefficient magnitudes and importance"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Data formatted for visualization components"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    model_info: Dict[str, Any] = Field(
        description="Additional model information and parameters"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metrics": {
                    "r2_score": 0.756,
                    "mse": 0.524,
                    "rmse": 0.724,
                    "mae": 0.532
                },
                "predictions": [2.45, 3.12, 1.87, 4.23, 2.98],
                "actual_values": [2.50, 3.00, 2.10, 4.10, 3.20],
                "coefficients": {
                    "MedInc": 0.829,
                    "HouseAge": 0.118,
                    "AveRooms": -0.265
                },
                "coefficient_analysis": {
                    "max_coefficient": {
                        "feature": "MedInc",
                        "value": 0.829
                    },
                    "min_coefficient": {
                        "feature": "Latitude",
                        "value": -0.045
                    },
                    "mean_coefficient_abs": 0.412
                },
                "visualization_data": {
                    "predictions_vs_actual": [
                        {"actual": 2.5, "predicted": 2.45, "index": 0},
                        {"actual": 3.0, "predicted": 3.12, "index": 1}
                    ],
                    "coefficient_plot": [
                        {"feature": "MedInc", "coefficient": 0.829},
                        {"feature": "HouseAge", "coefficient": 0.118}
                    ]
                },
                "execution_time_ms": 32.5,
                "model_info": {
                    "alpha": 1.0,
                    "fit_intercept": True,
                    "solver": "auto",
                    "n_features": 8,
                    "coef_l2_norm": 2.145
                }
            }
        }
