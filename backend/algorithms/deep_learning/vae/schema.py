"""Pydantic schemas for VAE API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class VAERequest(BaseModel):
    """Request schema for VAE training.

    Attributes:
        latent_dim: Dimension of latent space (default: 20, range: 2-64)
        encoder_hidden: List of hidden layer sizes for encoder (default: [128, 64])
        decoder_hidden: List of hidden layer sizes for decoder (default: [64, 128])
        learning_rate: Learning rate for optimizer (default: 0.001, range: 0.0001-0.01)
        epochs: Number of training epochs (default: 30, range: 10-100)
        beta: KL divergence weight (default: 1.0, range: 0.1-10.0)
        batch_size: Batch size for training (default: 128)
        random_state: Random seed for reproducibility (default: 42)
    """

    latent_dim: int = Field(
        default=20,
        ge=2,
        le=64,
        description="Dimension of latent space"
    )
    encoder_hidden: List[int] = Field(
        default=[128, 64],
        description="Hidden layer sizes for encoder"
    )
    decoder_hidden: List[int] = Field(
        default=[64, 128],
        description="Hidden layer sizes for decoder"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Learning rate for optimizer"
    )
    epochs: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Number of training epochs"
    )
    beta: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Weight for KL divergence term"
    )
    batch_size: int = Field(
        default=128,
        ge=16,
        le=512,
        description="Batch size for training"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "latent_dim": 20,
                "encoder_hidden": [128, 64],
                "decoder_hidden": [64, 128],
                "learning_rate": 0.001,
                "epochs": 30,
                "beta": 1.0,
                "batch_size": 128,
                "random_state": 42
            }
        }


class VAEResponse(BaseModel):
    """Response schema for VAE training results.

    Attributes:
        success: Whether the operation completed successfully
        metrics: Performance metrics (final losses, etc.)
        original_images: Original input images (sample)
        reconstructed_images: Reconstructed images (sample)
        generated_images: Images generated from random latent vectors
        latent_space: 2D projection of latent representations
        latent_labels: Class labels for latent space visualization
        loss_history: Training history of losses
        visualization_data: Data formatted for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        model_info: Information about the trained model
        parameters_used: Parameters that were used for training
        error: Error message if operation failed
    """

    success: bool
    metrics: Optional[Dict[str, float]] = None
    original_images: Optional[List[List[float]]] = None
    reconstructed_images: Optional[List[List[float]]] = None
    generated_images: Optional[List[List[float]]] = None
    latent_space: Optional[List[List[float]]] = None
    latent_labels: Optional[List[int]] = None
    loss_history: Optional[Dict[str, List[float]]] = None
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
                    "final_loss": 150.23,
                    "final_reconstruction_loss": 145.67,
                    "final_kl_divergence": 4.56
                },
                "execution_time_ms": 45230.5,
                "model_info": {
                    "latent_dim": 20,
                    "total_parameters": 156032,
                    "encoder_architecture": "[64, 128, 64]",
                    "decoder_architecture": "[64, 128, 64]"
                }
            }
        }
