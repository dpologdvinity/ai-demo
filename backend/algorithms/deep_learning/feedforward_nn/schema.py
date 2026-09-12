"""Pydantic schemas for MLP API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class MLPRequest(BaseModel):
    """Request schema for MLP training.

    Attributes:
        hidden_layers: Hidden layer sizes (e.g., [64, 32] for 2 hidden layers)
        learning_rate: Learning rate for optimizer
        epochs: Maximum number of training iterations
        batch_size: Size of minibatches for training
        activation: Activation function for hidden layers
        dataset_name: Name of dataset to use
        random_state: Random seed for reproducibility
    """

    hidden_layers: List[int] = Field(
        default=[64, 32],
        description="Sizes of hidden layers (e.g., [64, 32] for two hidden layers)"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.1,
        description="Learning rate for optimizer"
    )
    epochs: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of training iterations"
    )
    batch_size: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Size of minibatches for stochastic optimizers"
    )
    activation: str = Field(
        default="relu",
        description="Activation function for hidden layers"
    )
    dataset_name: str = Field(
        default="iris",
        description="Dataset to use for training"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    @field_validator('activation')
    @classmethod
    def validate_activation(cls, v: str) -> str:
        """Validate activation function choice."""
        allowed = ['relu', 'tanh', 'sigmoid', 'logistic']
        if v not in allowed:
            raise ValueError(f"activation must be one of {allowed}")
        return v

    @field_validator('hidden_layers')
    @classmethod
    def validate_hidden_layers(cls, v: List[int]) -> List[int]:
        """Validate hidden layer sizes."""
        if not v or len(v) == 0:
            raise ValueError("hidden_layers must contain at least one layer")
        if any(size <= 0 for size in v):
            raise ValueError("All hidden layer sizes must be positive")
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "hidden_layers": [64, 32],
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 32,
                "activation": "relu",
                "dataset_name": "iris",
                "random_state": 42
            }
        }


class MLPResponse(BaseModel):
    """Response schema for MLP training results.

    Attributes:
        success: Whether training was successful
        metrics: Performance metrics (accuracy, precision, recall, f1_score)
        predictions: Predicted labels for test set
        actual: Actual labels for test set
        training_history: Training curves (loss and accuracy per epoch)
        visualization_data: Data for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        model_info: Model architecture information
        parameters_used: Parameters used for training
    """

    success: bool = Field(
        description="Whether training completed successfully"
    )
    metrics: Dict[str, Any] = Field(
        description="Performance metrics including accuracy, precision, recall, f1_score"
    )
    predictions: List[int] = Field(
        description="Predicted class labels for test set"
    )
    actual: List[int] = Field(
        description="Actual class labels for test set"
    )
    training_history: Dict[str, List[float]] = Field(
        description="Training curves with loss and accuracy per iteration"
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
                    "accuracy": 0.967,
                    "precision": 0.969,
                    "recall": 0.967,
                    "f1_score": 0.967,
                    "confusion_matrix": [[10, 0, 0], [0, 9, 1], [0, 0, 10]]
                },
                "predictions": [0, 1, 2, 1, 0],
                "actual": [0, 1, 2, 1, 0],
                "training_history": {
                    "iterations": [1, 2, 3, 4, 5],
                    "loss": [1.2, 0.8, 0.5, 0.3, 0.2],
                    "accuracy": [0.5, 0.7, 0.85, 0.92, 0.97]
                },
                "visualization_data": {
                    "training_curves": {},
                    "confusion_matrix": [],
                    "network_architecture": {}
                },
                "execution_time_ms": 1234.56,
                "model_info": {
                    "hidden_layers": [64, 32],
                    "total_parameters": 5000
                },
                "parameters_used": {
                    "learning_rate": 0.001,
                    "epochs": 100
                }
            }
        }
