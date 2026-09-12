"""Pydantic schemas for Transfer Learning API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TransferLearningRequest(BaseModel):
    """Request schema for Transfer Learning training.

    Attributes:
        base_model: Pre-trained model architecture
        strategy: Transfer learning strategy
        freeze_layers: Which layers to freeze
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        dataset: Dataset to use for fine-tuning
        random_state: Random seed for reproducibility
    """

    base_model: str = Field(
        default="resnet18",
        description="Pre-trained model architecture",
        pattern="^(resnet18|resnet50|mobilenet_v2|efficientnet_b0)$"
    )
    strategy: str = Field(
        default="fine_tune",
        description="Transfer learning strategy",
        pattern="^(feature_extraction|fine_tune|full_train)$"
    )
    freeze_layers: str = Field(
        default="auto",
        description="Which layers to freeze",
        pattern="^(none|early|most|all_but_last|auto)$"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Learning rate for optimizer"
    )
    epochs: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        default=16,
        ge=4,
        le=64,
        description="Training batch size"
    )
    dataset: str = Field(
        default="flowers",
        description="Dataset for fine-tuning",
        pattern="^(flowers|animals|food)$"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "base_model": "resnet18",
                "strategy": "fine_tune",
                "freeze_layers": "auto",
                "learning_rate": 0.001,
                "epochs": 10,
                "batch_size": 16,
                "dataset": "flowers",
                "random_state": 42
            }
        }


class LayerInfo(BaseModel):
    """Information about a model layer."""

    name: str = Field(description="Layer name")
    trainable: bool = Field(description="Whether layer is trainable")
    num_params: int = Field(description="Number of parameters")
    frozen: bool = Field(description="Whether layer is frozen")


class StrategyComparison(BaseModel):
    """Comparison of different transfer learning strategies."""

    strategy: str = Field(description="Strategy name")
    final_accuracy: float = Field(description="Final validation accuracy")
    training_time_ms: float = Field(description="Training time in milliseconds")
    params_updated: int = Field(description="Number of parameters trained")
    convergence_epoch: int = Field(description="Epoch where model converged")


class TransferLearningResponse(BaseModel):
    """Response schema for Transfer Learning training results.

    Attributes:
        success: Whether training was successful
        metrics: Training and validation metrics
        training_history: Training curves data
        confusion_matrix: Confusion matrix for predictions
        layer_info: Information about layer freezing
        strategy_comparison: Comparison of different strategies
        feature_maps: Extracted features from frozen layers
        sample_predictions: Sample predictions with confidence
        visualization_data: Data for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        model_info: Model information and configuration
        parameters_used: Parameters used for training
    """

    success: bool = Field(
        description="Whether training completed successfully"
    )
    metrics: Dict[str, float] = Field(
        description="Training and validation metrics"
    )
    training_history: Dict[str, List[float]] = Field(
        description="Training curves (loss, accuracy)"
    )
    confusion_matrix: List[List[int]] = Field(
        description="Confusion matrix for predictions"
    )
    layer_info: List[LayerInfo] = Field(
        description="Information about each layer's training status"
    )
    strategy_comparison: Optional[List[StrategyComparison]] = Field(
        default=None,
        description="Comparison of different transfer learning strategies"
    )
    feature_maps: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted feature maps from frozen layers"
    )
    sample_predictions: List[Dict[str, Any]] = Field(
        description="Sample predictions with confidence scores"
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
                    "train_accuracy": 0.92,
                    "val_accuracy": 0.88,
                    "train_loss": 0.25,
                    "val_loss": 0.35,
                    "test_accuracy": 0.87
                },
                "training_history": {
                    "epochs": [1, 2, 3, 4, 5],
                    "train_loss": [1.2, 0.8, 0.5, 0.3, 0.25],
                    "train_accuracy": [0.6, 0.75, 0.85, 0.90, 0.92],
                    "val_loss": [1.0, 0.7, 0.5, 0.4, 0.35],
                    "val_accuracy": [0.65, 0.78, 0.83, 0.86, 0.88]
                },
                "confusion_matrix": [
                    [45, 2, 1],
                    [3, 42, 3],
                    [1, 2, 46]
                ],
                "layer_info": [
                    {"name": "conv1", "trainable": False, "num_params": 9408, "frozen": True},
                    {"name": "layer4", "trainable": True, "num_params": 147456, "frozen": False},
                    {"name": "fc", "trainable": True, "num_params": 2560, "frozen": False}
                ],
                "sample_predictions": [
                    {
                        "image_index": 0,
                        "true_label": "rose",
                        "predicted_label": "rose",
                        "confidence": 0.95,
                        "top_3_probs": [0.95, 0.03, 0.02]
                    }
                ],
                "visualization_data": {
                    "convergence_comparison": [],
                    "layer_freezing_diagram": {},
                    "feature_extraction_comparison": {}
                },
                "execution_time_ms": 45234.56,
                "model_info": {
                    "base_model": "resnet18",
                    "total_parameters": 11689512,
                    "trainable_parameters": 149632,
                    "frozen_parameters": 11539880,
                    "num_classes": 5
                },
                "parameters_used": {
                    "base_model": "resnet18",
                    "strategy": "fine_tune",
                    "freeze_layers": "auto",
                    "learning_rate": 0.001,
                    "epochs": 10
                }
            }
        }
