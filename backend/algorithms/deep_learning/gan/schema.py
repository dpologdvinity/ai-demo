"""Pydantic schemas for GAN API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GANRequest(BaseModel):
    """Request schema for GAN training.

    Attributes:
        latent_dim: Dimension of the noise vector input to generator.
            Default: 100, Range: 32-256
        g_hidden: Size of generator's hidden layer.
            Default: 256, Range: 128-512
        d_hidden: Size of discriminator's hidden layer.
            Default: 256, Range: 128-512
        learning_rate: Learning rate for both networks.
            Default: 0.0002, Range: 0.00001-0.001
        epochs: Number of training epochs.
            Default: 50, Range: 10-200
        batch_size: Batch size for training.
            Default: 64, Range: 16-256
        random_state: Random seed for reproducibility. Default: 42
    """

    latent_dim: int = Field(
        default=100,
        ge=32,
        le=256,
        description="Noise vector dimension"
    )
    g_hidden: int = Field(
        default=256,
        ge=128,
        le=512,
        description="Generator hidden layer size"
    )
    d_hidden: int = Field(
        default=256,
        ge=128,
        le=512,
        description="Discriminator hidden layer size"
    )
    learning_rate: float = Field(
        default=0.0002,
        ge=0.00001,
        le=0.001,
        description="Learning rate for Adam optimizer"
    )
    epochs: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of training epochs"
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
                "latent_dim": 100,
                "g_hidden": 128,
                "d_hidden": 128,
                "learning_rate": 0.0002,
                "epochs": 100,
                "batch_size": 64,
                "random_state": 42
            }
        }


class GANResponse(BaseModel):
    """Response schema for GAN training results.

    Attributes:
        loss_history: Training history with generator and discriminator losses
            - epoch: Epoch number
            - g_loss: Generator loss
            - d_loss: Discriminator loss
            - d_real_loss: Discriminator loss on real images
            - d_fake_loss: Discriminator loss on fake images
            - d_real_accuracy: Discriminator accuracy on real images
            - d_fake_accuracy: Discriminator accuracy on fake images
        generated_samples: Generated images at different epochs
            - epoch: Epoch number
            - samples: List of generated 8x8 images (flattened to 64 values)
        final_samples: Final generated samples after training
        interpolated_samples: Latent space interpolation samples (optional)
        decision_boundary: Discriminator decision boundary data (optional)
        visualization_data: Data for visualization
            - sample_epochs: Epochs at which samples were saved
            - loss_epochs: All epochs for loss plotting
            - n_samples: Number of training samples
            - image_shape: Shape of images [8, 8]
        execution_time_ms: Total training time in milliseconds
        model_info: Model configuration information
            - latent_dim: Latent dimension used
            - g_hidden: Generator hidden size
            - d_hidden: Discriminator hidden size
            - total_epochs: Total epochs trained
            - generator_params: Number of generator parameters
            - discriminator_params: Number of discriminator parameters
    """

    loss_history: List[Dict[str, float]] = Field(
        description="Training loss history for both networks"
    )
    generated_samples: List[Dict[str, Any]] = Field(
        description="Generated image samples at different epochs"
    )
    final_samples: List[List[float]] = Field(
        description="Final generated samples after training"
    )
    interpolated_samples: Optional[List[List[float]]] = Field(
        default=None,
        description="Latent space interpolation samples"
    )
    decision_boundary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Discriminator decision boundary visualization data"
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
                "loss_history": [
                    {
                        "epoch": 0,
                        "g_loss": 2.35,
                        "d_loss": 0.82,
                        "d_real_loss": 0.45,
                        "d_fake_loss": 0.37
                    },
                    {
                        "epoch": 1,
                        "g_loss": 2.12,
                        "d_loss": 0.75,
                        "d_real_loss": 0.42,
                        "d_fake_loss": 0.33
                    }
                ],
                "generated_samples": [
                    {
                        "epoch": 0,
                        "samples": [
                            [0.1, 0.2, 0.3],  # Flattened 8x8 image
                        ]
                    }
                ],
                "final_samples": [
                    [0.1, 0.2, 0.3],  # 16 samples of 64 values each
                ],
                "visualization_data": {
                    "sample_epochs": [0, 10, 20, 30],
                    "loss_epochs": [0, 1, 2, 3]
                },
                "execution_time_ms": 15423.5,
                "model_info": {
                    "latent_dim": 100,
                    "g_hidden": 128,
                    "d_hidden": 128,
                    "total_epochs": 100,
                    "generator_params": 20864,
                    "discriminator_params": 16641
                }
            }
        }
