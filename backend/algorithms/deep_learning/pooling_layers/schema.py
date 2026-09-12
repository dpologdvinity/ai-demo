"""Pydantic schemas for Pooling Layers API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class PoolingLayersRequest(BaseModel):
    """Request schema for pooling layers demonstration.

    Attributes:
        pool_type: Pooling method (max, average, global_max, global_average)
        pool_size: Size of the pooling window
        stride: Stride for pooling operation
        padding: Padding to add to input
        input_size: Size of input feature map (height, width)
        num_channels: Number of input channels
        random_state: Random seed for reproducibility
    """

    pool_type: str = Field(
        default='max',
        description="Pooling method to demonstrate"
    )
    pool_size: int = Field(
        default=2,
        ge=2,
        le=4,
        description="Size of the pooling window (pool_size x pool_size)"
    )
    stride: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Stride for pooling operation"
    )
    padding: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Padding to add to input feature map"
    )
    input_size: int = Field(
        default=8,
        ge=4,
        le=32,
        description="Size of square input feature map (input_size x input_size)"
    )
    num_channels: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Number of input channels"
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )

    @field_validator('pool_type')
    @classmethod
    def validate_pool_type(cls, v: str) -> str:
        """Validate pooling type choice."""
        allowed = ['max', 'average', 'global_max', 'global_average']
        if v not in allowed:
            raise ValueError(f"pool_type must be one of {allowed}")
        return v

    @field_validator('stride')
    @classmethod
    def validate_stride(cls, v: int) -> int:
        """Validate stride value."""
        if v < 1:
            raise ValueError("stride must be at least 1")
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "pool_type": "max",
                "pool_size": 2,
                "stride": 2,
                "padding": 0,
                "input_size": 8,
                "num_channels": 1,
                "random_state": 42
            }
        }


class PoolingLayersResponse(BaseModel):
    """Response schema for pooling layers demonstration.

    Attributes:
        success: Whether computation was successful
        input_feature_map: Original input feature map
        pooled_output: Output after pooling
        max_positions: Positions of maximum values (for max pooling)
        dimension_info: Input and output dimension information
        pooling_windows: Sample pooling windows with their operations
        comparison_data: Comparison of different pooling types on same input
        visualization_data: Formatted data for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        parameters_used: Parameters used for this computation
    """

    success: bool = Field(
        description="Whether computation completed successfully"
    )
    input_feature_map: List[List[List[float]]] = Field(
        description="Original input feature map [channels, height, width]"
    )
    pooled_output: List[List[List[float]]] = Field(
        description="Output after pooling [channels, output_h, output_w]"
    )
    max_positions: Optional[List[List[List[List[int]]]]] = Field(
        default=None,
        description="Positions of maximum values for max pooling [channels, out_h, out_w, [row, col]]"
    )
    dimension_info: Dict[str, Any] = Field(
        description="Input and output dimension information"
    )
    pooling_windows: List[Dict[str, Any]] = Field(
        description="Sample pooling windows showing the operation"
    )
    comparison_data: Dict[str, Any] = Field(
        description="Comparison of different pooling types on the same input"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Data formatted for frontend visualization"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    parameters_used: Dict[str, Any] = Field(
        description="Parameters used for this computation"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "input_feature_map": [[[1.0, 2.0], [3.0, 4.0]]],
                "pooled_output": [[[4.0]]],
                "max_positions": [[[[1, 1]]]],
                "dimension_info": {
                    "input_shape": [1, 2, 2],
                    "output_shape": [1, 1, 1],
                    "reduction_factor": 4
                },
                "pooling_windows": [],
                "comparison_data": {},
                "visualization_data": {},
                "execution_time_ms": 5.67,
                "parameters_used": {
                    "pool_type": "max",
                    "pool_size": 2,
                    "stride": 2
                }
            }
        }
