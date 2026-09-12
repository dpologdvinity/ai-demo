"""Request and response schemas for Sentiment Analysis algorithm."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class SentimentAnalysisParameters(BaseModel):
    """Parameters for Sentiment Analysis.

    Attributes:
        model_type: Model to use for sentiment analysis
        confidence_threshold: Minimum confidence score threshold
        neutral_threshold: Threshold for neutral classification (VADER compound score range)
        use_custom_texts: Whether to use custom texts for analysis
        custom_texts: Custom text samples for analysis
    """

    model_type: str = Field(
        default="vader",
        description="Model to use: vader, textblob, or transformers"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score threshold"
    )
    neutral_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Threshold for neutral classification (for VADER compound score)"
    )
    use_custom_texts: bool = Field(
        default=False,
        description="Whether to use custom texts"
    )
    custom_texts: Optional[List[str]] = Field(
        default=None,
        description="Custom text samples for analysis"
    )

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate that model_type is supported."""
        valid_models = ['vader', 'textblob', 'transformers']
        if v not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}")
        return v

    @field_validator('confidence_threshold')
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validate that confidence_threshold is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v


class SentimentPrediction(BaseModel):
    """Sentiment prediction for a single text.

    Attributes:
        text: The input text
        sentiment: Predicted sentiment label (positive, negative, neutral)
        confidence: Confidence score (0-1)
        scores: Detailed sentiment scores
        compound: Compound score (for VADER, -1 to 1)
    """
    text: str
    sentiment: str
    confidence: float
    scores: Dict[str, float] = Field(default_factory=dict)
    compound: Optional[float] = None


class SentimentDistribution(BaseModel):
    """Sentiment distribution statistics.

    Attributes:
        positive: Count of positive sentiments
        negative: Count of negative sentiments
        neutral: Count of neutral sentiments
        positive_pct: Percentage of positive sentiments
        negative_pct: Percentage of negative sentiments
        neutral_pct: Percentage of neutral sentiments
    """
    positive: int
    negative: int
    neutral: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float


class SentimentAnalysisResponse(BaseModel):
    """Response schema for Sentiment Analysis results.

    Attributes:
        success: Whether analysis completed successfully
        predictions: List of sentiment predictions for all texts
        distribution: Sentiment distribution statistics
        top_positive: Top N most positive texts
        top_negative: Top N most negative texts
        metrics: Analysis metrics including average confidence
        visualization_data: Data formatted for visualization
        execution_time_ms: Analysis execution time in milliseconds
        parameters_used: Actual parameters used for analysis
        model_info: Information about the model used
        error: Error message if analysis failed
    """

    success: bool
    predictions: List[SentimentPrediction] = Field(default_factory=list)
    distribution: Optional[SentimentDistribution] = None
    top_positive: List[SentimentPrediction] = Field(default_factory=list)
    top_negative: List[SentimentPrediction] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
