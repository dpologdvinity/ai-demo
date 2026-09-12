"""Pydantic schemas for ResNet API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ResNetRequest(BaseModel):
    """Request schema for ResNet inference.

    Attributes:
        model_variant: ResNet architecture variant
        top_k: Number of top predictions to return
        use_pretrained: Whether to use pre-trained weights
        image_path: Path to input image (optional, for file upload)
        image_index: Index of sample image from gallery (optional)
        random_state: Random seed for reproducibility
    """

    model_variant: str = Field(
        default="resnet18",
        description="ResNet architecture variant",
        pattern="^(resnet18|resnet34|resnet50|resnet101)$"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of top predictions to return"
    )
    use_pretrained: bool = Field(
        default=True,
        description="Whether to use pre-trained ImageNet weights"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Path to input image file"
    )
    image_index: Optional[int] = Field(
        default=0,
        ge=0,
        le=9,
        description="Index of sample image from gallery (0-9)"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "model_variant": "resnet18",
                "top_k": 5,
                "use_pretrained": True,
                "image_index": 0,
                "random_state": 42
            }
        }


class PredictionResult(BaseModel):
    """Single prediction result."""

    class_id: int = Field(description="ImageNet class ID")
    class_name: str = Field(description="Human-readable class name")
    confidence: float = Field(description="Prediction confidence (0-1)")


class ResidualBlockInfo(BaseModel):
    """Information about a residual block."""

    block_name: str = Field(description="Name/identifier of the block")
    input_channels: int = Field(description="Number of input channels")
    output_channels: int = Field(description="Number of output channels")
    stride: int = Field(description="Stride of the block")
    has_downsample: bool = Field(description="Whether block has downsampling")


class ResNetResponse(BaseModel):
    """Response schema for ResNet inference results.

    Attributes:
        success: Whether inference was successful
        predictions: Top-K predictions with confidence scores
        input_image: Input image data (normalized)
        input_shape: Shape of input image (H, W, C)
        feature_maps: Extracted feature maps from key layers
        visualization_data: Data for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        model_info: Model architecture information
        parameters_used: Parameters used for this inference
    """

    success: bool = Field(
        description="Whether inference completed successfully"
    )
    predictions: List[PredictionResult] = Field(
        description="Top-K predictions with class names and confidence scores"
    )
    input_image: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="Input image data (H, W, C) for visualization"
    )
    input_shape: List[int] = Field(
        description="Shape of input image [H, W, C]"
    )
    feature_maps: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted feature maps from key layers"
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
        description="Parameters used for this inference run"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "predictions": [
                    {"class_id": 281, "class_name": "tabby cat", "confidence": 0.45},
                    {"class_id": 282, "class_name": "tiger cat", "confidence": 0.32},
                    {"class_id": 285, "class_name": "Egyptian cat", "confidence": 0.12}
                ],
                "input_shape": [224, 224, 3],
                "visualization_data": {
                    "confidence_chart": [],
                    "residual_blocks": [],
                    "architecture_diagram": {}
                },
                "execution_time_ms": 234.56,
                "model_info": {
                    "model_variant": "resnet18",
                    "total_parameters": 11689512,
                    "depth": 18,
                    "num_residual_blocks": 8
                },
                "parameters_used": {
                    "model_variant": "resnet18",
                    "top_k": 5,
                    "use_pretrained": True
                }
            }
        }
