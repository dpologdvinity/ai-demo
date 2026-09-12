"""Pydantic schemas for XGBoost algorithm requests and responses."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator


class XGBoostRequest(BaseModel):
    """Request schema for XGBoost training.

    Attributes:
        n_estimators: Number of boosting rounds (trees to build).
        learning_rate: Step size shrinkage used to prevent overfitting.
        max_depth: Maximum tree depth for base learners.
        subsample: Subsample ratio of the training instances.
        dataset_name: Optional dataset name override.
        normalize: Whether to normalize features before training.
    """

    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of boosting rounds"
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Step size shrinkage"
    )
    max_depth: int = Field(
        default=6,
        ge=3,
        le=15,
        description="Maximum tree depth"
    )
    subsample: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Subsample ratio"
    )
    dataset_name: Optional[str] = Field(
        default="wine",
        description="Dataset to use for training"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize features"
    )

    @field_validator('n_estimators')
    @classmethod
    def validate_n_estimators(cls, v: int) -> int:
        """Validate n_estimators is in valid range."""
        if v < 10 or v > 500:
            raise ValueError('n_estimators must be between 10 and 500')
        return v

    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning_rate is in valid range."""
        if v < 0.01 or v > 1.0:
            raise ValueError('learning_rate must be between 0.01 and 1.0')
        return v

    @field_validator('max_depth')
    @classmethod
    def validate_max_depth(cls, v: int) -> int:
        """Validate max_depth is in valid range."""
        if v < 3 or v > 15:
            raise ValueError('max_depth must be between 3 and 15')
        return v

    @field_validator('subsample')
    @classmethod
    def validate_subsample(cls, v: float) -> float:
        """Validate subsample is in valid range."""
        if v < 0.5 or v > 1.0:
            raise ValueError('subsample must be between 0.5 and 1.0')
        return v


class XGBoostResponse(BaseModel):
    """Response schema for XGBoost training results.

    Attributes:
        success: Whether training completed successfully.
        metrics: Performance metrics including accuracy and F1 scores.
        predictions: Model predictions on test data.
        visualization_data: Data for frontend visualizations.
        execution_time_ms: Training execution time in milliseconds.
        parameters_used: Actual parameters used for training.
        error: Error message if training failed.
    """

    success: bool
    metrics: Dict[str, float]
    predictions: Optional[List[int]] = None
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any]
    error: Optional[str] = None
