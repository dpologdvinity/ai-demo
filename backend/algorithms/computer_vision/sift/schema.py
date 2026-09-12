"""SIFT schema definitions."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class SIFTRequest(BaseModel):
    """Request model for SIFT feature detection.

    Attributes:
        nfeatures: Maximum number of features to detect
        nOctaveLayers: Number of layers in each octave
        contrastThreshold: Contrast threshold for feature filtering
        edgeThreshold: Edge threshold for filtering edge-like features
        sigma: Gaussian sigma for the first octave
        image_index: Index of sample image to use (0-based)
        match_mode: Enable image pair matching mode
        match_image_index: Index of second image for matching
    """
    nfeatures: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum number of features to detect (more = slower but more detailed)"
    )
    nOctaveLayers: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of layers in each octave (scale levels)"
    )
    contrastThreshold: float = Field(
        default=0.04,
        ge=0.01,
        le=0.1,
        description="Contrast threshold for filtering weak features (higher = fewer features)"
    )
    edgeThreshold: float = Field(
        default=10,
        ge=5,
        le=20,
        description="Edge threshold for filtering edge-like features (higher = fewer edge features)"
    )
    sigma: float = Field(
        default=1.6,
        ge=0.5,
        le=3.0,
        description="Gaussian sigma for the first octave (smoothing level)"
    )
    image_index: int = Field(
        default=0,
        ge=0,
        description="Index of sample image to use"
    )
    match_mode: bool = Field(
        default=False,
        description="Enable matching mode (match features between two images)"
    )
    match_image_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Index of second image for matching (if match_mode is True)"
    )


class KeypointData(BaseModel):
    """Data for a single detected keypoint.

    Attributes:
        x: X coordinate
        y: Y coordinate
        size: Size/scale of the keypoint
        angle: Orientation angle in degrees
        response: Response strength
        octave: Octave index
    """
    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int


class SIFTStatistics(BaseModel):
    """Statistics about detected SIFT features.

    Attributes:
        keypoint_count: Number of keypoints detected
        image_dimensions: Width and height of the image
        average_scale: Average scale of keypoints
        average_response: Average response strength
        scale_distribution: Distribution of scales (histogram)
        orientation_distribution: Distribution of orientations (histogram)
        octave_distribution: Count of keypoints per octave
        descriptor_dimensions: Dimensions of feature descriptors (128)
    """
    keypoint_count: int
    image_dimensions: Dict[str, int]
    average_scale: float
    average_response: float
    scale_distribution: Dict[str, int] = Field(description="Scale histogram bins")
    orientation_distribution: Dict[str, int] = Field(description="Orientation histogram bins")
    octave_distribution: Dict[str, int] = Field(description="Keypoints per octave")
    descriptor_dimensions: int = Field(default=128, description="SIFT descriptor size")


class MatchStatistics(BaseModel):
    """Statistics about feature matching (if in match mode).

    Attributes:
        match_count: Number of good matches found
        total_matches: Total matches before filtering
        keypoints_image1: Keypoint count in first image
        keypoints_image2: Keypoint count in second image
        match_ratio: Ratio of matches to total keypoints
        average_match_distance: Average distance of good matches
    """
    match_count: int
    total_matches: int
    keypoints_image1: int
    keypoints_image2: int
    match_ratio: float
    average_match_distance: float


class SIFTResponse(BaseModel):
    """Response model for SIFT feature detection.

    Attributes:
        success: Whether feature detection was successful
        statistics: SIFT feature statistics
        visualization_data: Data for frontend visualization
        execution_time_ms: Time taken for feature detection in milliseconds
        algorithm_info: Information about the SIFT algorithm
        parameters_used: Parameters used for this detection
        image_info: Information about the processed image
        keypoints: List of detected keypoints (limited to top N for response size)
        match_statistics: Matching statistics (if in match mode)
    """
    success: bool = True
    statistics: SIFTStatistics
    visualization_data: Dict[str, Any]
    execution_time_ms: float
    algorithm_info: Dict[str, Any]
    parameters_used: Dict[str, Any]
    image_info: Dict[str, Any]
    keypoints: List[KeypointData] = Field(description="Sample of detected keypoints")
    match_statistics: Optional[MatchStatistics] = None
