"""Pydantic schemas for RNN API endpoints.

This module defines request and response models for the Recurrent Neural Network
algorithm endpoints, ensuring type safety and validation.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RNNRequest(BaseModel):
    """Request schema for RNN training.

    Attributes:
        hidden_size: Hidden state dimension
        num_layers: Number of RNN layers
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        sequence_length: Input sequence length for training
        prediction_length: Number of steps to predict ahead

    Example:
        >>> request = RNNRequest(
        ...     hidden_size=64,
        ...     num_layers=2,
        ...     learning_rate=0.001,
        ...     epochs=50,
        ...     sequence_length=20,
        ...     prediction_length=10
        ... )
    """

    hidden_size: int = Field(
        default=64,
        ge=32,
        le=256,
        description="Hidden state dimension"
    )
    num_layers: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of RNN layers"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Learning rate for optimizer"
    )
    epochs: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of training epochs"
    )
    sequence_length: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Input sequence length"
    )
    prediction_length: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of steps to predict ahead"
    )


class RNNResponse(BaseModel):
    """Response schema for RNN training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics (final loss, MSE)
        training_history: Loss values for each epoch
        predictions: Model predictions on test sequences
        actual: Actual target values from test sequences
        visualization_data: Data formatted for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        model_info: Information about the trained model
        parameters_used: Parameters that were used for training
        error: Error message if training failed

    Example:
        >>> response = RNNResponse(
        ...     success=True,
        ...     metrics={"final_loss": 0.015, "mse": 0.012},
        ...     training_history=[0.5, 0.3, 0.1, 0.05, 0.015],
        ...     predictions=[[1.2, 1.3, 1.4]],
        ...     actual=[[1.1, 1.3, 1.5]],
        ...     visualization_data={...},
        ...     execution_time_ms=5234.5,
        ...     model_info={...},
        ...     parameters_used={...}
        ... )
    """

    success: bool
    metrics: Dict[str, float] = Field(default_factory=dict)
    training_history: Optional[List[float]] = None
    predictions: Optional[List[List[float]]] = None
    actual: Optional[List[List[float]]] = None
    input_sequences: Optional[List[List[float]]] = None
    hidden_states: Optional[List[List[List[float]]]] = None
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
                    "final_loss": 0.015,
                    "mse": 0.012,
                    "mae": 0.095
                },
                "training_history": [0.5, 0.3, 0.15, 0.08, 0.05, 0.03, 0.015],
                "predictions": [[1.2, 1.3, 1.4, 1.5]],
                "actual": [[1.1, 1.3, 1.5, 1.4]],
                "input_sequences": [[0.5, 0.7, 0.9, 1.1]],
                "visualization_data": {
                    "training_loss": [
                        {"epoch": 1, "loss": 0.5},
                        {"epoch": 2, "loss": 0.3}
                    ],
                    "predictions_chart": [
                        {"step": 0, "predicted": 1.2, "actual": 1.1},
                        {"step": 1, "predicted": 1.3, "actual": 1.3}
                    ]
                },
                "execution_time_ms": 5234.5,
                "model_info": {
                    "input_size": 1,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "total_parameters": 12800
                },
                "parameters_used": {
                    "hidden_size": 64,
                    "num_layers": 2,
                    "learning_rate": 0.001,
                    "epochs": 50
                }
            }
        }
