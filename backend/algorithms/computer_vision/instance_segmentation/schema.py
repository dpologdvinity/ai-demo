"""Instance Segmentation (Mask R-CNN) schema definitions."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class InstanceSegmentationRequest(BaseModel):
    """Request model for instance segmentation.

    Attributes:
        confidence_threshold: Minimum confidence score for detections (0.1-0.9)
        model_backbone: Backbone network (resnet50 or resnet101)
        mask_threshold: Binary threshold for masks (0.1-0.9)
        max_instances: Maximum number of instances to detect (10-200)
        image_index: Index of sample image to use (0-9)
        nms_threshold: NMS IoU threshold (0.1-0.9)
    """
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Minimum confidence score for detections"
    )
    model_backbone: str = Field(
        default='resnet50',
        pattern='^(resnet50|resnet101)$',
        description="Backbone network: resnet50 or resnet101"
    )
    mask_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Mask binary threshold"
    )
    max_instances: int = Field(
        default=100,
        ge=10,
        le=200,
        description="Maximum number of instances to detect"
    )
    image_index: int = Field(
        default=0,
        ge=0,
        le=9,
        description="Index of sample image to use"
    )
    nms_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="NMS IoU threshold"
    )


class Instance(BaseModel):
    """Single instance detection result.

    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        class_name: Detected object class name
        class_id: Class ID in COCO dataset
        confidence: Detection confidence score (0-1)
        mask_area: Area of the segmentation mask in pixels
        instance_id: Unique instance identifier
    """
    bbox: List[float] = Field(description="Bounding box [x1, y1, x2, y2]")
    class_name: str = Field(description="Object class name")
    class_id: int = Field(description="COCO class ID")
    confidence: float = Field(description="Detection confidence")
    mask_area: int = Field(description="Mask area in pixels")
    instance_id: int = Field(description="Unique instance ID")


class InstanceStatistics(BaseModel):
    """Statistics about detected instances.

    Attributes:
        total_instances: Total number of instances detected
        unique_classes: Number of unique classes detected
        class_counts: Count of instances per class
        avg_confidence: Average confidence score
        total_mask_area: Total area covered by all masks
        coverage_percentage: Percentage of image covered by masks
        avg_instance_size: Average instance size in pixels
        confidence_distribution: Confidence scores bucketed into ranges
    """
    total_instances: int
    unique_classes: int
    class_counts: Dict[str, int]
    avg_confidence: float
    total_mask_area: int
    coverage_percentage: float
    avg_instance_size: float
    confidence_distribution: Dict[str, int] = Field(
        description="Instances grouped by confidence ranges"
    )


class InstanceSegmentationResponse(BaseModel):
    """Response model for instance segmentation.

    Attributes:
        success: Whether segmentation was successful
        instances: List of detected instances
        statistics: Instance statistics
        visualization_data: Data for frontend visualization
        execution_time_ms: Time taken for segmentation in milliseconds
        model_info: Information about the Mask R-CNN model used
        parameters_used: Parameters used for this segmentation
        image_info: Information about the processed image
    """
    success: bool = True
    instances: List[Instance]
    statistics: InstanceStatistics
    visualization_data: Dict[str, Any]
    execution_time_ms: float
    model_info: Dict[str, Any]
    parameters_used: Dict[str, Any]
    image_info: Dict[str, Any]
