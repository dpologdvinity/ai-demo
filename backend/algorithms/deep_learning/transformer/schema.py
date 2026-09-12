"""Pydantic schemas for Transformer API endpoints.

This module defines request and response models for the Transformer
algorithm endpoints, ensuring type safety and validation.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TransformerRequest(BaseModel):
    """Request schema for Transformer training.

    Attributes:
        d_model: Model dimension (dimensionality of embeddings/hidden states)
        nhead: Number of attention heads in multi-head attention
        num_layers: Number of encoder/decoder layers
        dim_feedforward: Dimension of feedforward network
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        dropout: Dropout rate for regularization

    Example:
        >>> request = TransformerRequest(
        ...     d_model=128,
        ...     nhead=8,
        ...     num_layers=2,
        ...     dim_feedforward=512,
        ...     learning_rate=0.001,
        ...     epochs=50,
        ...     dropout=0.1
        ... )
    """

    d_model: int = Field(
        default=128,
        ge=64,
        le=512,
        description="Model dimension (must be divisible by nhead)"
    )
    nhead: int = Field(
        default=8,
        description="Number of attention heads"
    )
    num_layers: int = Field(
        default=2,
        ge=1,
        le=6,
        description="Number of encoder/decoder layers"
    )
    dim_feedforward: int = Field(
        default=512,
        ge=256,
        le=2048,
        description="Dimension of feedforward network"
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
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class TransformerResponse(BaseModel):
    """Response schema for Transformer training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics (accuracy, loss)
        training_history: Loss values for each epoch
        predictions: Model predictions on test sequences
        actual: Actual target values from test sequences
        input_sequences: Input sequences used for prediction
        attention_weights: Attention weights for visualization
        visualization_data: Data formatted for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        model_info: Information about the trained model
        parameters_used: Parameters that were used for training
        error: Error message if training failed

    Example:
        >>> response = TransformerResponse(
        ...     success=True,
        ...     metrics={"accuracy": 0.95, "test_loss": 0.05},
        ...     training_history=[0.5, 0.3, 0.1, 0.05],
        ...     predictions=[[5, 4, 3, 2, 1]],
        ...     actual=[[5, 4, 3, 2, 1]],
        ...     visualization_data={...},
        ...     execution_time_ms=12500.0,
        ...     model_info={...},
        ...     parameters_used={...}
        ... )
    """

    success: bool
    metrics: Dict[str, float] = Field(default_factory=dict)
    training_history: Optional[List[float]] = None
    predictions: Optional[List[List[int]]] = None
    actual: Optional[List[List[int]]] = None
    input_sequences: Optional[List[List[int]]] = None
    attention_weights: Optional[List[List[List[float]]]] = None
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
                    "accuracy": 0.95,
                    "test_loss": 0.05,
                    "perplexity": 1.05
                },
                "training_history": [0.8, 0.5, 0.3, 0.2, 0.1, 0.05],
                "predictions": [[5, 4, 3, 2, 1]],
                "actual": [[5, 4, 3, 2, 1]],
                "input_sequences": [[1, 2, 3, 4, 5]],
                "attention_weights": [[[0.2, 0.3, 0.5]]],
                "visualization_data": {
                    "training_loss": [
                        {"epoch": 1, "loss": 0.8},
                        {"epoch": 2, "loss": 0.5}
                    ],
                    "attention_heatmap": {
                        "weights": [[0.2, 0.3, 0.5]],
                        "input_tokens": [1, 2, 3, 4, 5],
                        "output_tokens": [5, 4, 3, 2, 1]
                    },
                    "architecture": {
                        "encoder_layers": 2,
                        "decoder_layers": 2,
                        "attention_heads": 8
                    }
                },
                "execution_time_ms": 12500.0,
                "model_info": {
                    "d_model": 128,
                    "nhead": 8,
                    "num_layers": 2,
                    "total_parameters": 285440
                },
                "parameters_used": {
                    "d_model": 128,
                    "nhead": 8,
                    "num_layers": 2,
                    "learning_rate": 0.001,
                    "epochs": 50
                }
            }
        }
