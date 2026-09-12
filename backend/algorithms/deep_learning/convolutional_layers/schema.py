"""Pydantic schemas for Convolutional Layers API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ConvolutionalLayersRequest(BaseModel):
    """Request schema for Convolutional Layers demonstration.

    Attributes:
        num_filters: Number of convolutional filters to use
        kernel_size: Size of the convolution kernel
        stride: Stride for convolution operation
        padding: Padding type ('same' or 'valid')
        activation: Activation function to apply
        random_state: Random seed for reproducibility
    """

    num_filters: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Number of convolutional filters"
    )
    kernel_size: int = Field(
        default=3,
        description="Size of convolution kernel (3, 5, or 7)"
    )
    stride: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Stride for convolution operation"
    )
    padding: str = Field(
        default='same',
        description="Padding type: 'same' or 'valid'"
    )
    activation: str = Field(
        default='relu',
        description="Activation function: 'relu', 'tanh', or 'none'"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "num_filters": 32,
                "kernel_size": 3,
                "stride": 1,
                "padding": "same",
                "activation": "relu",
                "random_state": 42
            }
        }


class ConvolutionalLayersResponse(BaseModel):
    """Response schema for Convolutional Layers demonstration.

    Attributes:
        success: Whether operation was successful
        input_image: Original input image
        filter_kernels: Visualization of filter weights
        feature_maps: Output feature maps after convolution
        common_filters: Predefined filters (Sobel, Gaussian, etc.)
        output_dimensions: Dimensions after convolution
        activation_stats: Statistics about activations
        visualization_data: Data for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        model_info: Model configuration information
        parameters_used: Parameters used for this demonstration
    """

    success: bool = Field(
        description="Whether operation completed successfully"
    )
    input_image: List[List[float]] = Field(
        description="Original input image as 2D array"
    )
    filter_kernels: List[Dict[str, Any]] = Field(
        description="Filter kernels with weights and visualizations"
    )
    feature_maps: List[Dict[str, Any]] = Field(
        description="Output feature maps for each filter"
    )
    common_filters: Dict[str, Any] = Field(
        description="Predefined filters (Sobel, Gaussian, etc.) applied to input"
    )
    output_dimensions: Dict[str, int] = Field(
        description="Output dimensions after convolution"
    )
    activation_stats: Dict[str, Any] = Field(
        description="Statistics about activation patterns"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Data formatted for frontend visualization"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    model_info: Dict[str, Any] = Field(
        description="Model configuration and details"
    )
    parameters_used: Dict[str, Any] = Field(
        description="Parameters used for this demonstration"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "input_image": [[0.1, 0.2], [0.3, 0.4]],
                "filter_kernels": [
                    {
                        "filter_id": 0,
                        "weights": [[0.1, 0.2], [0.3, 0.4]]
                    }
                ],
                "feature_maps": [
                    {
                        "filter_id": 0,
                        "output": [[0.5, 0.6], [0.7, 0.8]]
                    }
                ],
                "common_filters": {
                    "sobel_x": [[0.5, 0.6]],
                    "sobel_y": [[0.5, 0.6]]
                },
                "output_dimensions": {
                    "height": 8,
                    "width": 8,
                    "channels": 32
                },
                "activation_stats": {
                    "mean_activation": 0.5,
                    "max_activation": 1.0
                },
                "visualization_data": {},
                "execution_time_ms": 123.45,
                "model_info": {},
                "parameters_used": {}
            }
        }
