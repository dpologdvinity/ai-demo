"""Pydantic schemas for CNN API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CNNRequest(BaseModel):
    """Request schema for CNN training.

    Attributes:
        conv_filters: Number of filters per convolutional layer
        kernel_size: Size of convolution kernel
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        dropout: Dropout rate for regularization
        batch_size: Batch size for training
        dataset_name: Name of dataset to use
        random_state: Random seed for reproducibility
    """

    conv_filters: List[int] = Field(
        default=[16, 32],
        description="Number of filters per convolutional layer"
    )
    kernel_size: int = Field(
        default=3,
        ge=3,
        le=7,
        description="Size of convolution kernel (odd number)"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Learning rate for Adam optimizer"
    )
    epochs: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Number of training epochs"
    )
    dropout: float = Field(
        default=0.5,
        ge=0.0,
        le=0.8,
        description="Dropout rate for regularization"
    )
    batch_size: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Batch size for training"
    )
    dataset_name: str = Field(
        default="digits",
        description="Dataset to use for training"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "conv_filters": [16, 32],
                "kernel_size": 3,
                "learning_rate": 0.001,
                "epochs": 15,
                "dropout": 0.5,
                "batch_size": 32,
                "dataset_name": "digits",
                "random_state": 42
            }
        }


class CNNResponse(BaseModel):
    """Response schema for CNN training results.

    Attributes:
        success: Whether training was successful
        metrics: Performance metrics (accuracy, precision, recall, f1_score)
        predictions: Predicted labels for test set
        actual: Actual labels for test set
        training_history: Training curves (loss and accuracy per epoch)
        feature_maps: Feature maps from first convolutional layer
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
        description="Training curves with loss and accuracy per epoch"
    )
    feature_maps: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="Feature maps from first convolutional layer for visualization"
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
                    "accuracy": 0.982,
                    "precision": 0.983,
                    "recall": 0.982,
                    "f1_score": 0.982,
                    "confusion_matrix": [[18, 0], [1, 17]]
                },
                "predictions": [0, 1, 2, 3, 4, 5],
                "actual": [0, 1, 2, 3, 4, 5],
                "training_history": {
                    "train_loss": [1.5, 0.8, 0.5],
                    "train_accuracy": [0.6, 0.8, 0.9],
                    "val_loss": [1.4, 0.7, 0.5],
                    "val_accuracy": [0.65, 0.82, 0.91]
                },
                "visualization_data": {
                    "training_curves": [],
                    "confusion_matrix": [],
                    "network_architecture": {}
                },
                "execution_time_ms": 12345.67,
                "model_info": {
                    "conv_filters": [16, 32],
                    "total_parameters": 50000
                },
                "parameters_used": {
                    "learning_rate": 0.001,
                    "epochs": 15
                }
            }
        }
