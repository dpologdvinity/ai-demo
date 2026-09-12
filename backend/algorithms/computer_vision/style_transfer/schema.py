"""Neural Style Transfer schema definitions."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class StyleTransferRequest(BaseModel):
    """Request model for neural style transfer.

    Attributes:
        content_image_index: Index of content image (0-9)
        style_image_index: Index of style image (0-9)
        iterations: Number of optimization steps (50-1000)
        content_weight: Weight for content loss (0.1-10.0)
        style_weight: Weight for style loss (100000-10000000)
        learning_rate: Optimizer learning rate (0.001-0.01)
        image_size: Output image size in pixels (256-1024)
    """
    content_image_index: int = Field(
        default=0,
        ge=0,
        le=9,
        description="Content image index (0-9)"
    )
    style_image_index: int = Field(
        default=0,
        ge=0,
        le=9,
        description="Style image index (0-9)"
    )
    iterations: int = Field(
        default=300,
        ge=50,
        le=1000,
        description="Number of optimization iterations"
    )
    content_weight: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Content loss weight"
    )
    style_weight: float = Field(
        default=1000000.0,
        ge=100000.0,
        le=10000000.0,
        description="Style loss weight"
    )
    learning_rate: float = Field(
        default=0.003,
        ge=0.001,
        le=0.01,
        description="Optimizer learning rate"
    )
    image_size: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Output image size"
    )


class LossHistory(BaseModel):
    """Loss values during optimization.

    Attributes:
        iteration: Iteration number
        total_loss: Total loss value
        content_loss: Content loss value
        style_loss: Style loss value
    """
    iteration: int
    total_loss: float
    content_loss: float
    style_loss: float


class FeatureVisualization(BaseModel):
    """Feature map visualization data.

    Attributes:
        layer_name: Name of the VGG layer
        feature_map: Base64 encoded feature map image
        description: Description of what this layer captures
    """
    layer_name: str
    feature_map: str
    description: str


class StyleTransferStatistics(BaseModel):
    """Statistics from style transfer process.

    Attributes:
        total_iterations: Total number of iterations completed
        final_total_loss: Final total loss value
        final_content_loss: Final content loss value
        final_style_loss: Final style loss value
        initial_total_loss: Initial total loss value
        loss_reduction: Percentage loss reduction
        convergence_rate: Rate of convergence
    """
    total_iterations: int
    final_total_loss: float
    final_content_loss: float
    final_style_loss: float
    initial_total_loss: float
    loss_reduction: float
    convergence_rate: float


class StyleTransferResponse(BaseModel):
    """Response model for neural style transfer.

    Attributes:
        success: Whether style transfer was successful
        statistics: Statistics from optimization process
        visualization_data: Data for frontend visualization
        loss_history: Loss values over iterations
        feature_visualizations: Feature map visualizations
        execution_time_ms: Time taken for style transfer in milliseconds
        model_info: Information about the VGG model used
        parameters_used: Parameters used for this style transfer
        image_info: Information about content and style images
    """
    success: bool = True
    statistics: StyleTransferStatistics
    visualization_data: Dict[str, Any]
    loss_history: List[LossHistory]
    feature_visualizations: Optional[List[FeatureVisualization]] = None
    execution_time_ms: float
    model_info: Dict[str, Any]
    parameters_used: Dict[str, Any]
    image_info: Dict[str, Any]
