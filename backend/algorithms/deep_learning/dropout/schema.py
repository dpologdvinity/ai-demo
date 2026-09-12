"""Pydantic schemas for Dropout API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class DropoutRequest(BaseModel):
    """Request schema for Dropout regularization demonstration.

    Attributes:
        dropout_rate: Probability of dropping neurons (0.0 to 0.9)
        apply_to_layers: Which layers get dropout
        training_epochs: Training duration
        hidden_layers: Hidden layer sizes for the network
        learning_rate: Learning rate for optimizer
        batch_size: Size of minibatches for training
        dataset_name: Name of dataset to use
        random_state: Random seed for reproducibility
    """

    dropout_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=0.9,
        description="Probability of dropping neurons during training"
    )
    apply_to_layers: List[str] = Field(
        default=["hidden1", "hidden2"],
        description="Which layers get dropout applied"
    )
    training_epochs: int = Field(
        default=100,
        ge=20,
        le=300,
        description="Number of training epochs"
    )
    hidden_layers: List[int] = Field(
        default=[128, 64],
        description="Sizes of hidden layers in the network"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.1,
        description="Learning rate for optimizer"
    )
    batch_size: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Size of minibatches for training"
    )
    dataset_name: str = Field(
        default="overfitting_demo",
        description="Dataset to use for training"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

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
                "dropout_rate": 0.5,
                "apply_to_layers": ["hidden1", "hidden2"],
                "training_epochs": 100,
                "hidden_layers": [128, 64],
                "learning_rate": 0.001,
                "batch_size": 32,
                "dataset_name": "overfitting_demo",
                "random_state": 42
            }
        }


class DropoutResponse(BaseModel):
    """Response schema for Dropout demonstration results.

    Attributes:
        success: Whether training was successful
        metrics: Performance metrics for all dropout rates
        training_curves: Training and validation loss curves
        dropout_comparison: Comparison of different dropout rates
        overfitting_metrics: Overfitting gap measurements
        dropout_masks: Visualization of dropout masks
        visualization_data: Data for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        model_info: Model architecture information
        parameters_used: Parameters used for training
    """

    success: bool = Field(
        description="Whether training completed successfully"
    )
    metrics: Dict[str, Any] = Field(
        description="Performance metrics for all dropout configurations"
    )
    training_curves: Dict[str, Any] = Field(
        description="Training and validation loss curves for different dropout rates"
    )
    dropout_comparison: List[Dict[str, Any]] = Field(
        description="Comparison of models with different dropout rates"
    )
    overfitting_metrics: Dict[str, Any] = Field(
        description="Overfitting gap (train loss - val loss) measurements"
    )
    dropout_masks: List[Dict[str, Any]] = Field(
        description="Visualization of which neurons are active/inactive during dropout"
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
                    "no_dropout": {
                        "train_accuracy": 0.99,
                        "val_accuracy": 0.75,
                        "overfitting_gap": 0.24
                    },
                    "dropout_0.5": {
                        "train_accuracy": 0.88,
                        "val_accuracy": 0.86,
                        "overfitting_gap": 0.02
                    }
                },
                "training_curves": {
                    "epochs": [1, 2, 3],
                    "no_dropout_train": [0.5, 0.2, 0.1],
                    "no_dropout_val": [0.6, 0.5, 0.4],
                    "dropout_train": [0.4, 0.3, 0.25],
                    "dropout_val": [0.38, 0.28, 0.24]
                },
                "dropout_comparison": [],
                "overfitting_metrics": {},
                "dropout_masks": [],
                "visualization_data": {},
                "execution_time_ms": 1234.56,
                "model_info": {},
                "parameters_used": {}
            }
        }
