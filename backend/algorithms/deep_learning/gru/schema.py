"""Pydantic schemas for GRU algorithm."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class GRURequest(BaseModel):
    """Request schema for GRU training."""

    hidden_size: int = Field(
        default=64,
        ge=32,
        le=256,
        description="Hidden layer size"
    )
    num_layers: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Number of GRU layers"
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
        description="Training epochs"
    )
    sequence_length: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Input sequence length"
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Dropout rate between layers"
    )
    batch_size: int = Field(
        default=32,
        ge=16,
        le=128,
        description="Training batch size"
    )
    n_samples: int = Field(
        default=1000,
        ge=500,
        le=5000,
        description="Total number of time steps to generate"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class GRUMetrics(BaseModel):
    """Metrics for GRU model evaluation."""

    train_loss: float = Field(description="Final training loss (MSE)")
    test_loss: float = Field(description="Test loss (MSE)")
    train_mse: float = Field(description="Training Mean Squared Error")
    test_mse: float = Field(description="Test Mean Squared Error")
    train_mae: float = Field(description="Training Mean Absolute Error")
    test_mae: float = Field(description="Test Mean Absolute Error")
    train_r2: float = Field(description="Training R² score")
    test_r2: float = Field(description="Test R² score")


class GateActivations(BaseModel):
    """Gate activations for visualization."""

    update_gate: List[List[float]] = Field(description="Update gate activations over time")
    reset_gate: List[List[float]] = Field(description="Reset gate activations over time")
    hidden_state: List[List[float]] = Field(description="Hidden state evolution over time")
    time_steps: List[int] = Field(description="Time step indices")


class VisualizationData(BaseModel):
    """Visualization data for GRU model."""

    training_curves: Dict[str, List[float]] = Field(
        description="Training and validation loss curves"
    )
    predictions: List[Dict[str, float]] = Field(
        description="Predictions vs actual values on test set"
    )
    gate_activations: Optional[GateActivations] = Field(
        default=None,
        description="Gate activations for sample sequence"
    )
    sequence_visualization: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Sample input sequence with prediction"
    )


class ModelInfo(BaseModel):
    """Information about the trained model."""

    total_parameters: int = Field(description="Total number of trainable parameters")
    hidden_size: int = Field(description="Hidden state dimension")
    num_layers: int = Field(description="Number of GRU layers")
    dropout: float = Field(description="Dropout rate")
    sequence_length: int = Field(description="Input sequence length")
    device: str = Field(description="Device used for training (cpu or cuda)")


class GRUResponse(BaseModel):
    """Response schema for GRU training."""

    success: bool = Field(default=True, description="Whether training succeeded")
    metrics: GRUMetrics = Field(description="Model performance metrics")
    visualization_data: VisualizationData = Field(description="Data for visualization")
    execution_time_ms: float = Field(description="Total execution time in milliseconds")
    model_info: ModelInfo = Field(description="Model architecture information")
    parameters_used: Dict[str, Any] = Field(description="Parameters used for training")
    error: Optional[str] = Field(default=None, description="Error message if training failed")
