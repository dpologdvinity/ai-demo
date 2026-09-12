"""Pydantic schemas for Autoencoder Variants API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AutoencoderVariantsRequest(BaseModel):
    """Request schema for Autoencoder Variants training.

    Attributes:
        variant: Autoencoder type to use
        latent_dim: Latent space dimension
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        noise_factor: Noise level for denoising variant
        sparsity_weight: Sparsity penalty weight
        batch_size: Training batch size
        random_state: Random seed for reproducibility
    """

    variant: str = Field(
        default='vanilla',
        description="Autoencoder variant: 'vanilla', 'denoising', 'sparse', or 'contractive'"
    )
    latent_dim: int = Field(
        default=32,
        ge=2,
        le=128,
        description="Latent space dimension for dimensionality reduction"
    )
    epochs: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Learning rate for Adam optimizer"
    )
    noise_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=0.5,
        description="Noise level for denoising autoencoder (0.0-0.5)"
    )
    sparsity_weight: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Weight for sparsity penalty (L1 regularization)"
    )
    batch_size: int = Field(
        default=128,
        ge=32,
        le=256,
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
                "variant": "vanilla",
                "latent_dim": 32,
                "epochs": 10,
                "learning_rate": 0.001,
                "noise_factor": 0.3,
                "sparsity_weight": 0.001,
                "batch_size": 128,
                "random_state": 42
            }
        }


class AutoencoderVariantsResponse(BaseModel):
    """Response schema for Autoencoder Variants training results.

    Attributes:
        success: Whether training completed successfully
        variant: The variant that was trained
        metrics: Performance metrics
        loss_history: Training loss over epochs
        original_images: Sample original images
        noisy_images: Noisy input images (for denoising variant)
        reconstructed_images: Reconstructed images
        latent_space: 2D latent space coordinates
        latent_labels: Labels for latent space points
        reconstruction_errors: Per-sample reconstruction error
        learned_filters: First layer encoder filters
        visualization_data: Additional visualization data
        execution_time_ms: Training time in milliseconds
        model_info: Model architecture information
        parameters_used: Parameters used for training
    """

    success: bool = Field(description="Whether training completed successfully")
    variant: str = Field(description="Autoencoder variant used")
    metrics: Dict[str, float] = Field(description="Performance metrics")
    loss_history: List[Dict[str, float]] = Field(
        description="Training history with loss per epoch"
    )
    original_images: List[List[float]] = Field(
        description="Sample of original test images"
    )
    noisy_images: Optional[List[List[float]]] = Field(
        default=None,
        description="Noisy input images (for denoising variant)"
    )
    reconstructed_images: List[List[float]] = Field(
        description="Reconstructed images"
    )
    latent_space: List[List[float]] = Field(
        description="2D latent space coordinates for visualization"
    )
    latent_labels: List[int] = Field(
        description="Labels for latent space points"
    )
    reconstruction_errors: List[float] = Field(
        description="Per-sample reconstruction error (MSE)"
    )
    learned_filters: List[List[float]] = Field(
        description="First layer encoder filters visualization"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Data for creating visualizations"
    )
    execution_time_ms: float = Field(
        description="Total training time in milliseconds"
    )
    model_info: Dict[str, Any] = Field(
        description="Model configuration and architecture information"
    )
    parameters_used: Dict[str, Any] = Field(
        description="Parameters used for training"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "variant": "vanilla",
                "metrics": {
                    "final_loss": 0.042,
                    "reconstruction_mse": 0.040,
                    "avg_reconstruction_error": 0.035
                },
                "loss_history": [
                    {"epoch": 0, "train_loss": 0.15, "val_loss": 0.14},
                    {"epoch": 1, "train_loss": 0.08, "val_loss": 0.09}
                ],
                "original_images": [[0.1, 0.2]],
                "noisy_images": None,
                "reconstructed_images": [[0.12, 0.19]],
                "latent_space": [[0.5, 1.2], [-0.3, 0.8]],
                "latent_labels": [0, 1],
                "reconstruction_errors": [0.035, 0.042],
                "learned_filters": [[0.1, 0.2]],
                "visualization_data": {
                    "n_samples": 200,
                    "image_shape": [28, 28],
                    "latent_dim": 32
                },
                "execution_time_ms": 8532.5,
                "model_info": {
                    "variant": "vanilla",
                    "latent_dim": 32,
                    "total_params": 100000
                },
                "parameters_used": {
                    "variant": "vanilla",
                    "latent_dim": 32,
                    "epochs": 10
                }
            }
        }
