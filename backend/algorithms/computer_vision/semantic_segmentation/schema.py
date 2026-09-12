"""Semantic Segmentation schema definitions."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SegmentationRequest(BaseModel):
    """Request model for semantic segmentation.

    Attributes:
        num_classes: Number of segmentation classes (2-150)
        confidence_threshold: Minimum confidence for predictions (0.1-0.95)
        model_backbone: Model backbone architecture
        image_size: Input image size
        image_index: Index of sample image to use (0-based)
    """
    num_classes: int = Field(
        default=21,
        ge=2,
        le=150,
        description="Number of segmentation classes"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.95,
        description="Minimum confidence for predictions"
    )
    model_backbone: str = Field(
        default='resnet50',
        pattern='^(resnet50|mobilenet)$',
        description="Backbone architecture: resnet50 or mobilenet"
    )
    image_size: int = Field(
        default=512,
        description="Input image size"
    )
    image_index: int = Field(
        default=0,
        ge=0,
        description="Index of sample image to use"
    )


class ClassInfo(BaseModel):
    """Information about a segmented class.

    Attributes:
        class_id: Class identifier
        class_name: Name of the class
        color: RGB color for visualization [r, g, b]
        pixel_count: Number of pixels in this class
        percentage: Percentage of image pixels
        iou_score: Intersection over Union score (if ground truth available)
    """
    class_id: int = Field(description="Class identifier")
    class_name: str = Field(description="Class name")
    color: List[int] = Field(description="RGB color [r, g, b]")
    pixel_count: int = Field(description="Number of pixels")
    percentage: float = Field(description="Percentage of image")
    iou_score: Optional[float] = Field(
        default=None,
        description="IoU score if ground truth available"
    )


class SegmentationStatistics(BaseModel):
    """Statistics about segmentation results.

    Attributes:
        total_classes: Total number of classes present
        total_pixels: Total number of pixels
        mean_confidence: Mean prediction confidence
        class_info: Information per class
    """
    total_classes: int
    total_pixels: int
    mean_confidence: float
    class_info: List[ClassInfo]


class SegmentationResponse(BaseModel):
    """Response model for semantic segmentation.

    Attributes:
        success: Whether segmentation was successful
        statistics: Segmentation statistics
        visualization_data: Data for frontend visualization
        execution_time_ms: Time taken for segmentation in milliseconds
        model_info: Information about the model used
        parameters_used: Parameters used for this segmentation
        image_info: Information about the processed image
    """
    success: bool = True
    statistics: SegmentationStatistics
    visualization_data: Dict[str, Any]
    execution_time_ms: float
    model_info: Dict[str, Any]
    parameters_used: Dict[str, Any]
    image_info: Dict[str, Any]
