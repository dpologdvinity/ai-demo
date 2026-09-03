"""
Standardized response schemas for API endpoints.

This module defines Pydantic models for consistent request/response
structures across all algorithm endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AlgorithmInfoResponse(BaseModel):
    """Response schema for algorithm information endpoints.

    Provides metadata and available datasets for a specific algorithm.

    Attributes:
        metadata: Complete algorithm metadata including parameters,
                 complexity, theory, and configuration
        available_datasets: List of dataset names that can be used
                          with this algorithm

    Example:
        >>> response = AlgorithmInfoResponse(
        ...     metadata={
        ...         "name": "Linear Regression",
        ...         "slug": "linear-regression",
        ...         "parameters": [...]
        ...     },
        ...     available_datasets=["boston", "california"]
        ... )
    """

    metadata: Dict[str, Any]
    available_datasets: List[str]


class TrainingRequest(BaseModel):
    """Generic request schema for algorithm training.

    Provides a standardized interface for configuring and executing
    algorithm training across different algorithm types.

    Attributes:
        parameters: Algorithm-specific parameters (e.g., learning_rate,
                   max_iterations, etc.)
        dataset_name: Optional dataset name override. If not provided,
                     uses algorithm's default dataset
        normalize: Whether to normalize/scale input features before training.
                  Defaults to True.

    Example:
        >>> request = TrainingRequest(
        ...     parameters={
        ...         "learning_rate": 0.01,
        ...         "max_iterations": 1000
        ...     },
        ...     dataset_name="boston",
        ...     normalize=True
        ... )
    """

    parameters: Dict[str, Any]
    dataset_name: Optional[str] = None
    normalize: bool = True


class TrainingResponse(BaseModel):
    """Generic response schema for algorithm training results.

    Provides comprehensive training results including metrics,
    predictions, visualizations, and execution metadata.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics (e.g., accuracy, loss, R2 score)
        predictions: Model predictions on test/validation data
        visualization_data: Data formatted for frontend visualization
                          (chart data, confusion matrices, etc.)
        execution_time_ms: Training execution time in milliseconds
        parameters_used: Actual parameters used for training
                        (includes defaults and overrides)
        error: Error message if training failed

    Example:
        >>> response = TrainingResponse(
        ...     success=True,
        ...     metrics={"r2_score": 0.85, "mse": 0.12},
        ...     predictions=[1.2, 3.4, 5.6],
        ...     visualization_data={
        ...         "x": [1, 2, 3],
        ...         "y_true": [1.1, 3.3, 5.5],
        ...         "y_pred": [1.2, 3.4, 5.6]
        ...     },
        ...     execution_time_ms=45.3,
        ...     parameters_used={"learning_rate": 0.01}
        ... )
    """

    success: bool
    metrics: Dict[str, float]
    predictions: Optional[List[Any]] = None
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any]
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response schema.

    Provides consistent error information across all API endpoints.

    Attributes:
        error: High-level error message suitable for display
        detail: Optional detailed error information for debugging
        status_code: HTTP status code (default: 500)

    Example:
        >>> error = ErrorResponse(
        ...     error="Invalid parameter",
        ...     detail="learning_rate must be between 0 and 1",
        ...     status_code=400
        ... )
    """

    error: str
    detail: Optional[str] = None
    status_code: int = 500
