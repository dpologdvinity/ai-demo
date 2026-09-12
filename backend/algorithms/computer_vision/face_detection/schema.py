"""Face Detection schema definitions."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List


class FaceDetectionRequest(BaseModel):
    """Request model for Face Detection using Haar Cascades.

    Attributes:
        scale_factor: Scale reduction factor between successive scans (1.05-1.3)
        min_neighbors: Minimum neighbors required for detection (1-10)
        min_size: Minimum face size in pixels (20-100)
        max_size: Maximum face size in pixels (None for image size)
        image_index: Index of sample image to use (0-based)
    """
    scale_factor: float = Field(
        default=1.1,
        ge=1.05,
        le=1.3,
        description="Scale reduction factor between successive scans (lower = slower but more accurate)"
    )
    min_neighbors: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Minimum neighbors required for detection (higher = fewer false positives)"
    )
    min_size: int = Field(
        default=30,
        ge=20,
        le=100,
        description="Minimum face size in pixels"
    )
    max_size: int | None = Field(
        default=None,
        ge=50,
        description="Maximum face size in pixels (None for no limit)"
    )
    image_index: int = Field(
        default=0,
        ge=0,
        description="Index of sample image to use"
    )


class FaceBox(BaseModel):
    """Detected face bounding box.

    Attributes:
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of the bounding box
        height: Height of the bounding box
        confidence: Detection confidence (higher neighbors = higher confidence)
        center: Center point of the face
        area: Area of the face in pixels
    """
    x: int
    y: int
    width: int
    height: int
    confidence: float = Field(description="Normalized confidence based on neighbors")
    center: Dict[str, int]
    area: int


class FaceStatistics(BaseModel):
    """Statistics about detected faces.

    Attributes:
        face_count: Total number of faces detected
        average_face_size: Average face size in pixels
        largest_face_size: Size of the largest detected face
        smallest_face_size: Size of the smallest detected face
        total_face_area: Total area covered by faces
        face_density: Percentage of image covered by faces
        image_dimensions: Width and height of the image
        detection_quality: Quality rating based on parameters
    """
    face_count: int
    average_face_size: float
    largest_face_size: int
    smallest_face_size: int
    total_face_area: int
    face_density: float = Field(description="Percentage of image covered by faces")
    image_dimensions: Dict[str, int]
    detection_quality: str = Field(description="Quality rating: High/Medium/Low")


class FaceDetectionResponse(BaseModel):
    """Response model for Face Detection.

    Attributes:
        success: Whether face detection was successful
        faces: List of detected face bounding boxes
        statistics: Face detection statistics
        visualization_data: Data for frontend visualization
        execution_time_ms: Time taken for face detection in milliseconds
        algorithm_info: Information about the Haar Cascade algorithm
        parameters_used: Parameters used for this detection
        image_info: Information about the processed image
    """
    success: bool = True
    faces: List[FaceBox]
    statistics: FaceStatistics
    visualization_data: Dict[str, Any]
    execution_time_ms: float
    algorithm_info: Dict[str, Any]
    parameters_used: Dict[str, Any]
    image_info: Dict[str, Any]
