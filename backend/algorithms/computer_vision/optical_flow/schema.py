"""Optical Flow schema definitions."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class OpticalFlowRequest(BaseModel):
    """Request model for Optical Flow estimation.

    Attributes:
        method: Flow algorithm ('farneback' or 'lucas-kanade')
        pyr_scale: Pyramid scale factor (0.3-0.9)
        levels: Number of pyramid levels (1-5)
        winsize: Window size for flow calculation (5-50)
        iterations: Number of iterations (1-10)
        image_pair_index: Index of sample image pair to use (0-based)
    """
    method: str = Field(
        default="farneback",
        description="Flow algorithm to use (farneback or lucas-kanade)"
    )
    pyr_scale: float = Field(
        default=0.5,
        ge=0.3,
        le=0.9,
        description="Pyramid scale factor - specifies image scale at each pyramid level"
    )
    levels: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of pyramid levels - more levels detect larger motions"
    )
    winsize: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Window size for flow calculation - larger windows are more robust to noise"
    )
    iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of iterations at each pyramid level"
    )
    image_pair_index: int = Field(
        default=0,
        ge=0,
        description="Index of sample image pair to use"
    )

    def validate_method(self):
        """Validate that method is one of the supported algorithms."""
        if self.method not in ["farneback", "lucas-kanade"]:
            raise ValueError("method must be 'farneback' or 'lucas-kanade'")

    def validate_winsize(self):
        """Validate that winsize is odd."""
        if self.winsize % 2 == 0:
            raise ValueError("winsize must be odd")


class FlowStatistics(BaseModel):
    """Statistics about computed optical flow.

    Attributes:
        average_magnitude: Average motion magnitude across all pixels
        max_magnitude: Maximum motion magnitude
        min_magnitude: Minimum motion magnitude
        median_magnitude: Median motion magnitude
        flow_coverage: Percentage of pixels with significant motion (>threshold)
        primary_direction: Primary direction of motion in degrees (0-360)
        image_dimensions: Width and height of the frames
        total_pixels: Total number of pixels analyzed
    """
    average_magnitude: float
    max_magnitude: float
    min_magnitude: float
    median_magnitude: float
    flow_coverage: float = Field(description="Percentage of pixels with significant motion")
    primary_direction: float = Field(description="Primary motion direction in degrees")
    image_dimensions: Dict[str, int]
    total_pixels: int


class FlowVector(BaseModel):
    """Individual flow vector for visualization.

    Attributes:
        x: X coordinate of vector origin
        y: Y coordinate of vector origin
        dx: X component of motion
        dy: Y component of motion
        magnitude: Vector magnitude
        angle: Vector angle in degrees
    """
    x: int
    y: int
    dx: float
    dy: float
    magnitude: float
    angle: float


class OpticalFlowResponse(BaseModel):
    """Response model for Optical Flow estimation.

    Attributes:
        success: Whether flow estimation was successful
        statistics: Flow computation statistics
        visualization_data: Data for frontend visualization
        flow_vectors: Sample flow vectors for arrow visualization
        execution_time_ms: Time taken for flow estimation in milliseconds
        algorithm_info: Information about the optical flow algorithm
        parameters_used: Parameters used for this computation
        image_info: Information about the processed images
    """
    success: bool = True
    statistics: FlowStatistics
    visualization_data: Dict[str, Any]
    flow_vectors: List[FlowVector]
    execution_time_ms: float
    algorithm_info: Dict[str, Any]
    parameters_used: Dict[str, Any]
    image_info: Dict[str, Any]
