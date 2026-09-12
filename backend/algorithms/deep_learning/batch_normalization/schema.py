"""Pydantic schemas for Batch Normalization API requests and responses."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class BatchNormRequest(BaseModel):
    """Request schema for Batch Normalization demonstration.

    Attributes:
        momentum: Running stats momentum for batch normalization
        eps: Numerical stability epsilon
        affine: Whether to use learnable scale and shift parameters
        track_running_stats: Whether to track running mean and variance
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        hidden_size: Size of hidden layers in the network
        random_state: Random seed for reproducibility
    """

    momentum: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Momentum for running statistics (default: 0.1)"
    )
    eps: float = Field(
        default=1e-5,
        ge=1e-8,
        le=1e-3,
        description="Small constant for numerical stability (default: 1e-5)"
    )
    affine: bool = Field(
        default=True,
        description="Whether to use learnable affine parameters (scale and shift)"
    )
    track_running_stats: bool = Field(
        default=True,
        description="Whether to track running mean and variance statistics"
    )
    epochs: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=0.01,
        ge=0.0001,
        le=0.1,
        description="Learning rate for optimizer"
    )
    batch_size: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Training batch size"
    )
    hidden_size: int = Field(
        default=64,
        ge=32,
        le=256,
        description="Size of hidden layers in the network"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "momentum": 0.1,
                "eps": 1e-5,
                "affine": True,
                "track_running_stats": True,
                "epochs": 50,
                "learning_rate": 0.01,
                "batch_size": 32,
                "hidden_size": 64,
                "random_state": 42
            }
        }


class BatchNormResponse(BaseModel):
    """Response schema for Batch Normalization training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Comparison metrics between models with and without BN
        training_curves: Training loss curves for both models
        convergence_comparison: Epochs to reach convergence threshold
        activation_distributions: Activation statistics per layer over time
        gradient_flow: Gradient norm statistics for both models
        visualization_data: Data formatted for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        model_info: Model architecture information
        parameters_used: Parameters used for training
    """

    success: bool = Field(
        description="Whether training completed successfully"
    )
    metrics: Dict[str, Any] = Field(
        description="Comparison metrics between BN and non-BN models"
    )
    training_curves: Dict[str, List[float]] = Field(
        description="Training loss curves for models with and without BN"
    )
    convergence_comparison: Dict[str, Any] = Field(
        description="Convergence analysis (epochs to reach threshold, final loss)"
    )
    activation_distributions: Dict[str, Any] = Field(
        description="Activation distribution statistics across layers and epochs"
    )
    gradient_flow: Dict[str, Any] = Field(
        description="Gradient norm comparison between BN and non-BN models"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Data formatted for frontend visualization"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    model_info: Dict[str, Any] = Field(
        description="Model architecture and configuration details"
    )
    parameters_used: Dict[str, Any] = Field(
        description="Parameters used for this training run"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "metrics": {
                    "with_bn_final_loss": 0.123,
                    "without_bn_final_loss": 0.456,
                    "improvement_percent": 72.8,
                    "with_bn_epochs_to_converge": 25,
                    "without_bn_epochs_to_converge": 45
                },
                "training_curves": {
                    "epochs": [1, 2, 3],
                    "with_bn_loss": [0.5, 0.3, 0.1],
                    "without_bn_loss": [0.6, 0.5, 0.4]
                },
                "convergence_comparison": {},
                "activation_distributions": {},
                "gradient_flow": {},
                "visualization_data": {},
                "execution_time_ms": 2500.0,
                "model_info": {},
                "parameters_used": {}
            }
        }
