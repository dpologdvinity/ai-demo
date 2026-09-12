"""Pydantic schemas for Learning Rate Scheduling API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class LearningRateSchedulingRequest(BaseModel):
    """Request schema for learning rate scheduling demonstration.

    Attributes:
        schedule_type: Type of learning rate schedule to use
        initial_lr: Starting learning rate
        step_size: Number of epochs between LR decay (for step schedule)
        gamma: Multiplicative factor of learning rate decay
        epochs: Number of training epochs
        min_lr: Minimum learning rate (for cosine schedule)
        max_lr: Maximum learning rate (for cyclic schedule)
        patience: Number of epochs to wait for improvement (for plateau schedule)
        random_state: Random seed for reproducibility
    """

    schedule_type: str = Field(
        default='step',
        description="Type of learning rate schedule"
    )
    initial_lr: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Starting learning rate"
    )
    step_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Steps before decay (for step schedule)"
    )
    gamma: float = Field(
        default=0.1,
        ge=0.01,
        le=0.9,
        description="Decay factor for learning rate"
    )
    epochs: int = Field(
        default=100,
        ge=20,
        le=300,
        description="Number of training epochs"
    )
    min_lr: float = Field(
        default=0.0001,
        ge=0.0,
        le=0.1,
        description="Minimum learning rate (for cosine schedule)"
    )
    max_lr: float = Field(
        default=0.5,
        ge=0.01,
        le=2.0,
        description="Maximum learning rate (for cyclic schedule)"
    )
    patience: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Patience for ReduceLROnPlateau"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    @field_validator('schedule_type')
    @classmethod
    def validate_schedule_type(cls, v: str) -> str:
        """Validate learning rate schedule choice."""
        allowed = ['step', 'exponential', 'cosine', 'reduce_on_plateau', 'cyclic']
        if v not in allowed:
            raise ValueError(f"schedule_type must be one of {allowed}")
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "schedule_type": "step",
                "initial_lr": 0.1,
                "step_size": 10,
                "gamma": 0.1,
                "epochs": 100,
                "min_lr": 0.0001,
                "max_lr": 0.5,
                "patience": 5,
                "random_state": 42
            }
        }


class LearningRateSchedulingResponse(BaseModel):
    """Response schema for learning rate scheduling demonstration.

    Attributes:
        success: Whether training was successful
        schedule_data: Learning rate values for each schedule over epochs
        loss_data: Training loss for each schedule over epochs
        convergence_metrics: Convergence metrics for each schedule
        comparison_table: Comparison of final performance across schedules
        visualization_data: Data formatted for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        parameters_used: Parameters used for this training
    """

    success: bool = Field(
        description="Whether training completed successfully"
    )
    schedule_data: Dict[str, List[float]] = Field(
        description="Learning rate values for each schedule over epochs"
    )
    loss_data: Dict[str, List[float]] = Field(
        description="Training loss for each schedule over epochs"
    )
    convergence_metrics: Dict[str, Dict[str, Any]] = Field(
        description="Convergence metrics for each schedule"
    )
    comparison_table: List[Dict[str, Any]] = Field(
        description="Comparison table of final performance across schedules"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Data formatted for frontend visualization"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    parameters_used: Dict[str, Any] = Field(
        description="Parameters used for this training"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "schedule_data": {
                    "step": [0.1, 0.1, 0.01, 0.01, 0.001],
                    "exponential": [0.1, 0.09, 0.081, 0.073, 0.066]
                },
                "loss_data": {
                    "step": [2.5, 1.8, 1.2, 0.8, 0.5],
                    "exponential": [2.5, 1.7, 1.1, 0.7, 0.4]
                },
                "convergence_metrics": {
                    "step": {
                        "final_loss": 0.5,
                        "convergence_epoch": 45,
                        "training_time": 123.4
                    }
                },
                "comparison_table": [
                    {
                        "schedule": "Step",
                        "final_loss": 0.5,
                        "convergence_epoch": 45,
                        "training_time": 123.4
                    }
                ],
                "visualization_data": {},
                "execution_time_ms": 1234.56,
                "parameters_used": {
                    "schedule_type": "step",
                    "initial_lr": 0.1,
                    "epochs": 100
                }
            }
        }
