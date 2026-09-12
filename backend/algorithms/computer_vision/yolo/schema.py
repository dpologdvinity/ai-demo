"""YOLO Object Detection schema definitions."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class YOLORequest(BaseModel):
    """Request model for YOLO object detection.

    Attributes:
        confidence_threshold: Minimum confidence score for detections (0.1-0.9)
        iou_threshold: IoU threshold for Non-Maximum Suppression (0.1-0.9)
        model_version: YOLO model version (yolov8n, yolov8s, yolov8m, yolov5s)
        max_detections: Maximum number of objects to detect (10-300)
        image_index: Index of sample image to use (0-14)
        class_filter: Filter detections by class category (all, person, vehicle, animal)
    """
    confidence_threshold: float = Field(
        default=0.25,
        ge=0.1,
        le=0.9,
        description="Minimum confidence score for detections"
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.1,
        le=0.9,
        description="IoU threshold for Non-Maximum Suppression"
    )
    model_version: str = Field(
        default='yolov8n',
        pattern='^(yolov8n|yolov8s|yolov8m|yolov5s)$',
        description="YOLO model version"
    )
    max_detections: int = Field(
        default=100,
        ge=10,
        le=300,
        description="Maximum number of objects to detect"
    )
    image_index: int = Field(
        default=0,
        ge=0,
        le=14,
        description="Index of sample image to use (0-14)"
    )
    class_filter: str = Field(
        default='all',
        pattern='^(all|person|vehicle|animal)$',
        description="Filter detections by class category"
    )

    # Legacy support - map old model_size to model_version
    @property
    def model_size(self) -> str:
        """Extract model size from model_version for backward compatibility."""
        if self.model_version.startswith('yolov8'):
            return self.model_version[-1]  # 'n', 's', or 'm'
        elif self.model_version == 'yolov5s':
            return 's'
        return 'n'


class Detection(BaseModel):
    """Single object detection result.

    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        class_name: Detected object class name
        class_id: Class ID in COCO dataset
        confidence: Detection confidence score (0-1)
    """
    bbox: List[float] = Field(description="Bounding box [x1, y1, x2, y2]")
    class_name: str = Field(description="Object class name")
    class_id: int = Field(description="COCO class ID")
    confidence: float = Field(description="Detection confidence")


class DetectionStatistics(BaseModel):
    """Statistics about detected objects.

    Attributes:
        total_detections: Total number of objects detected
        class_counts: Count of objects per class
        avg_confidence: Average confidence score
        confidence_distribution: Confidence scores bucketed into ranges
    """
    total_detections: int
    class_counts: Dict[str, int]
    avg_confidence: float
    confidence_distribution: Dict[str, int] = Field(
        description="Detections grouped by confidence ranges"
    )


class YOLOResponse(BaseModel):
    """Response model for YOLO object detection.

    Attributes:
        success: Whether detection was successful
        detections: List of detected objects
        statistics: Detection statistics
        visualization_data: Data for frontend visualization
        execution_time_ms: Time taken for detection in milliseconds
        model_info: Information about the YOLO model used
        parameters_used: Parameters used for this detection
        image_info: Information about the processed image
    """
    success: bool = True
    detections: List[Detection]
    statistics: DetectionStatistics
    visualization_data: Dict[str, Any]
    execution_time_ms: float
    model_info: Dict[str, Any]
    parameters_used: Dict[str, Any]
    image_info: Dict[str, Any]
