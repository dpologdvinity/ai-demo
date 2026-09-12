"""Edge Detection schema definitions."""

from pydantic import BaseModel, Field
from typing import Dict, Any


class EdgeDetectionRequest(BaseModel):
    """Request model for Edge Detection using Canny algorithm.

    Attributes:
        threshold1: Lower threshold for hysteresis (0-255)
        threshold2: Upper threshold for hysteresis (0-255)
        aperture_size: Sobel kernel size (3, 5, or 7)
        l2gradient: Use L2 norm for gradient magnitude (default: False)
        image_index: Index of sample image to use (0-based)
    """
    threshold1: int = Field(
        default=50,
        ge=0,
        le=255,
        description="Lower threshold for hysteresis (edges with gradient below this are discarded)"
    )
    threshold2: int = Field(
        default=150,
        ge=0,
        le=255,
        description="Upper threshold for hysteresis (edges with gradient above this are kept)"
    )
    aperture_size: int = Field(
        default=3,
        ge=3,
        le=7,
        description="Sobel kernel size (must be 3, 5, or 7)"
    )
    l2gradient: bool = Field(
        default=False,
        description="Use L2 norm for gradient magnitude calculation (more accurate but slower)"
    )
    image_index: int = Field(
        default=0,
        ge=0,
        description="Index of sample image to use"
    )

    def validate_aperture_size(self):
        """Validate that aperture_size is odd and in range."""
        if self.aperture_size not in [3, 5, 7]:
            raise ValueError("aperture_size must be 3, 5, or 7")


class EdgeStatistics(BaseModel):
    """Statistics about detected edges.

    Attributes:
        edge_pixel_count: Number of pixels classified as edges
        total_pixels: Total number of pixels in the image
        edge_density: Percentage of pixels that are edges (0-100)
        image_dimensions: Width and height of the image
        threshold_ratio: Ratio of threshold2 to threshold1
    """
    edge_pixel_count: int
    total_pixels: int
    edge_density: float = Field(description="Percentage of edge pixels (0-100)")
    image_dimensions: Dict[str, int] = Field(description="Image width and height")
    threshold_ratio: float = Field(description="Ratio of upper to lower threshold")


class EdgeDetectionResponse(BaseModel):
    """Response model for Edge Detection.

    Attributes:
        success: Whether edge detection was successful
        statistics: Edge detection statistics
        visualization_data: Data for frontend visualization
        execution_time_ms: Time taken for edge detection in milliseconds
        algorithm_info: Information about the Canny algorithm
        parameters_used: Parameters used for this detection
        image_info: Information about the processed image
    """
    success: bool = True
    statistics: EdgeStatistics
    visualization_data: Dict[str, Any]
    execution_time_ms: float
    algorithm_info: Dict[str, Any]
    parameters_used: Dict[str, Any]
    image_info: Dict[str, Any]
