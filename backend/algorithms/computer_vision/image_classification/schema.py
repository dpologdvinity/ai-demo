"""Image Classification schema definitions."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ImageClassificationRequest(BaseModel):
    """Request model for image classification.

    Attributes:
        model_name: Pre-trained model to use (resnet18, resnet50, mobilenet_v2)
        top_k: Number of top predictions to return (1-10)
        confidence_threshold: Minimum confidence for predictions (0.0-1.0)
        image_index: Index of sample image to use (0-based)
    """
    model_name: str = Field(
        default='resnet18',
        pattern='^(resnet18|resnet50|mobilenet_v2)$',
        description="Pre-trained model: resnet18, resnet50, or mobilenet_v2"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of top predictions to return"
    )
    confidence_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for predictions"
    )
    image_index: int = Field(
        default=0,
        ge=0,
        description="Index of sample image to use"
    )


class Prediction(BaseModel):
    """Single prediction result.

    Attributes:
        class_name: Predicted class name
        class_id: Class ID in ImageNet
        confidence: Prediction confidence score (0-1)
        probability: Probability percentage (0-100)
    """
    class_name: str = Field(description="Predicted class name")
    class_id: int = Field(description="ImageNet class ID")
    confidence: float = Field(description="Prediction confidence (0-1)")
    probability: float = Field(description="Probability percentage (0-100)")


class PredictionStatistics(BaseModel):
    """Statistics about predictions.

    Attributes:
        total_predictions: Number of predictions returned
        top_confidence: Highest confidence score
        confidence_spread: Difference between top and bottom confidence
        entropy: Prediction entropy (uncertainty measure)
    """
    total_predictions: int
    top_confidence: float
    confidence_spread: float
    entropy: float


class ImageClassificationResponse(BaseModel):
    """Response model for image classification.

    Attributes:
        success: Whether classification was successful
        predictions: List of top-K predictions
        statistics: Prediction statistics
        visualization_data: Data for frontend visualization
        execution_time_ms: Time taken for classification in milliseconds
        model_info: Information about the model used
        parameters_used: Parameters used for this classification
        image_info: Information about the processed image
    """
    success: bool = True
    predictions: List[Prediction]
    statistics: PredictionStatistics
    visualization_data: Dict[str, Any]
    execution_time_ms: float
    model_info: Dict[str, Any]
    parameters_used: Dict[str, Any]
    image_info: Dict[str, Any]
