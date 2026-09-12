"""Pydantic schemas for Autoencoder API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AutoencoderRequest(BaseModel):
    """Request schema for Autoencoder training.

    Attributes:
        latent_dim: Dimension of the latent space.
            Default: 32, Range: 2-128
        hidden_dim: Size of hidden layers in encoder/decoder.
            Default: 128, Range: 64-512
        epochs: Number of training epochs.
            Default: 50, Range: 10-200
        learning_rate: Learning rate for Adam optimizer.
            Default: 0.001, Range: 0.0001-0.01
        batch_size: Batch size for training.
            Default: 64, Range: 16-256
        random_state: Random seed for reproducibility. Default: 42
    """

    latent_dim: int = Field(
        default=32,
        ge=2,
        le=128,
        description="Latent space dimension for dimensionality reduction"
    )
    hidden_dim: int = Field(
        default=128,
        ge=64,
        le=512,
        description="Hidden layer size in encoder and decoder"
    )
    epochs: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Learning rate for Adam optimizer"
    )
    batch_size: int = Field(
        default=64,
        ge=16,
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
                "latent_dim": 32,
                "hidden_dim": 128,
                "epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 64,
                "random_state": 42
            }
        }


class AutoencoderResponse(BaseModel):
    """Response schema for Autoencoder training results.

    Attributes:
        reconstruction_loss: Final reconstruction loss (MSE) on test set
        avg_pixel_error: Average per-pixel reconstruction error
        loss_history: Training history with reconstruction loss per epoch
            - epoch: Epoch number
            - train_loss: Training reconstruction loss
            - val_loss: Validation reconstruction loss
        original_images: Sample of original test images for comparison
        reconstructed_images: Corresponding reconstructed images
        latent_representations: 2D latent space coordinates for visualization
            (if latent_dim=2) or t-SNE projection (if latent_dim>2)
        visualization_data: Data for creating visualizations
            - n_samples: Number of samples used
            - image_shape: Shape of images [8, 8]
            - latent_dim: Latent dimension used
            - projection_method: 'none' if latent_dim=2, 'tsne' otherwise
        execution_time_ms: Total training time in milliseconds
        model_info: Model configuration information
            - latent_dim: Latent dimension used
            - hidden_dim: Hidden dimension used
            - total_epochs: Total epochs trained
            - encoder_params: Number of encoder parameters
            - decoder_params: Number of decoder parameters
    """

    reconstruction_loss: float = Field(
        description="Final MSE reconstruction loss on test set"
    )
    avg_pixel_error: float = Field(
        description="Average per-pixel reconstruction error"
    )
    loss_history: List[Dict[str, float]] = Field(
        description="Training history with loss per epoch"
    )
    original_images: List[List[float]] = Field(
        description="Sample of original test images (5-10 examples)"
    )
    reconstructed_images: List[List[float]] = Field(
        description="Corresponding reconstructed images"
    )
    latent_representations: List[Dict[str, float]] = Field(
        description="2D coordinates for latent space visualization"
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

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "reconstruction_loss": 0.045,
                "avg_pixel_error": 0.021,
                "loss_history": [
                    {"epoch": 0, "train_loss": 0.15, "val_loss": 0.14},
                    {"epoch": 1, "train_loss": 0.08, "val_loss": 0.09}
                ],
                "original_images": [[0.1, 0.2, 0.3]],  # 64 values each
                "reconstructed_images": [[0.12, 0.19, 0.31]],
                "latent_representations": [
                    {"x": 0.5, "y": 1.2, "label": 0},
                    {"x": -0.3, "y": 0.8, "label": 1}
                ],
                "visualization_data": {
                    "n_samples": 200,
                    "image_shape": [8, 8],
                    "latent_dim": 32,
                    "projection_method": "tsne"
                },
                "execution_time_ms": 8532.5,
                "model_info": {
                    "latent_dim": 32,
                    "hidden_dim": 128,
                    "total_epochs": 50,
                    "encoder_params": 12345,
                    "decoder_params": 12345
                }
            }
        }
