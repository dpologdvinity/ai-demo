"""Pydantic schemas for Naive Bayes algorithm requests and responses."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class NaiveBayesRequest(BaseModel):
    """Request schema for Naive Bayes training.

    Attributes:
        var_smoothing: Portion of the largest variance of all features
            that is added to variances for calculation stability.
            Must be between 1e-10 and 1e-8.
        priors: Prior probabilities of the classes. If specified,
            must sum to 1. If None, priors are adjusted according to the data.
        dataset_name: Name of dataset to use (default: 'iris')
        normalize: Whether to normalize features before training (default: True)
    """

    var_smoothing: float = Field(
        default=1e-9,
        ge=1e-10,
        le=1e-8,
        description="Variance smoothing parameter"
    )
    priors: Optional[List[float]] = Field(
        default=None,
        description="Prior probabilities of classes"
    )
    dataset_name: str = Field(
        default="iris",
        description="Dataset to use for training"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize features"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "var_smoothing": 1e-9,
                "priors": None,
                "dataset_name": "iris",
                "normalize": True
            }
        }


class NaiveBayesResponse(BaseModel):
    """Response schema for Naive Bayes training results.

    Attributes:
        success: Whether training was successful
        metrics: Performance metrics (accuracy, precision, recall, f1)
        predictions: Model predictions on test set
        probabilities: Class probability distributions for test samples
        feature_contributions: Contribution of each feature to predictions
        visualization_data: Data for visualization (confusion matrix, probability dist)
        execution_time_ms: Time taken for training in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed
    """

    success: bool
    metrics: Dict[str, float] = Field(default_factory=dict)
    predictions: Optional[List[int]] = None
    probabilities: Optional[List[List[float]]] = None
    feature_contributions: Optional[Dict[str, Any]] = None
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "metrics": {
                    "accuracy": 0.95,
                    "precision": 0.94,
                    "recall": 0.95,
                    "f1_score": 0.94
                },
                "predictions": [0, 1, 2, 1, 0],
                "probabilities": [
                    [0.9, 0.05, 0.05],
                    [0.1, 0.85, 0.05]
                ],
                "visualization_data": {
                    "confusion_matrix": [[10, 0, 0], [0, 9, 1], [0, 1, 9]],
                    "class_names": ["setosa", "versicolor", "virginica"]
                },
                "execution_time_ms": 12.5,
                "parameters_used": {
                    "var_smoothing": 1e-9,
                    "priors": None
                }
            }
        }
