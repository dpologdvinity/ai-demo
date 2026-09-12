"""
Pydantic schemas for K-Nearest Neighbors algorithm.

This module defines request and response models for the KNN algorithm,
including parameter validation and response structure.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class KNNRequest(BaseModel):
    """Request model for KNN training and prediction.

    Attributes:
        n_neighbors: Number of neighbors to use (K value)
        weights: Weight function used in prediction
        metric: Distance metric for neighbor calculation
        p: Power parameter for Minkowski metric
        test_size: Proportion of dataset to use for testing
        random_state: Random seed for reproducibility
    """

    n_neighbors: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of neighbors K to consider"
    )
    weights: Literal['uniform', 'distance'] = Field(
        default='uniform',
        description="Weight function: 'uniform' (equal) or 'distance' (weighted by inverse distance)"
    )
    metric: Literal['euclidean', 'manhattan', 'minkowski'] = Field(
        default='euclidean',
        description="Distance metric to use for finding neighbors"
    )
    p: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Power parameter for Minkowski metric (p=1: Manhattan, p=2: Euclidean)"
    )
    test_size: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="Proportion of data to use for testing"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducible splits"
    )


class KNNMetrics(BaseModel):
    """Classification metrics from KNN model.

    Attributes:
        accuracy: Overall classification accuracy (0-1)
        precision: Precision score (weighted average)
        recall: Recall score (weighted average)
        f1_score: F1 score (weighted average)
        train_accuracy: Accuracy on training set
        test_accuracy: Accuracy on test set
    """

    accuracy: float = Field(description="Overall classification accuracy")
    precision: float = Field(description="Precision (weighted average)")
    recall: float = Field(description="Recall (weighted average)")
    f1_score: float = Field(description="F1 score (weighted average)")
    train_accuracy: float = Field(description="Training set accuracy")
    test_accuracy: float = Field(description="Test set accuracy")


class VisualizationData(BaseModel):
    """Visualization data for KNN results.

    Attributes:
        scatter_data: 2D projection of data points with predictions
        confusion_matrix: Confusion matrix for classification results
        class_labels: Names of the target classes
        decision_boundary: Optional decision boundary data for 2D visualization
    """

    scatter_data: List[Dict[str, Any]] = Field(
        description="Data points with x, y coordinates and class labels"
    )
    confusion_matrix: List[List[int]] = Field(
        description="Confusion matrix showing prediction vs actual"
    )
    class_labels: List[str] = Field(
        description="Names of the classification classes"
    )
    decision_boundary: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Decision boundary visualization data (if 2D)"
    )


class KNNResponse(BaseModel):
    """Response model for KNN training results.

    Attributes:
        success: Whether the training was successful
        metrics: Classification performance metrics
        predictions: List of predicted class labels for test set
        actual: List of actual class labels for test set
        visualization_data: Data for frontend visualization
        execution_time_ms: Time taken for training and prediction in milliseconds
        parameters: The parameters used for this training run
        message: Optional message about the training
    """

    success: bool = Field(description="Whether training succeeded")
    metrics: KNNMetrics = Field(description="Performance metrics")
    predictions: List[int] = Field(description="Predicted labels")
    actual: List[int] = Field(description="Actual labels")
    visualization_data: VisualizationData = Field(description="Visualization data")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    parameters: Dict[str, Any] = Field(description="Parameters used")
    message: Optional[str] = Field(default=None, description="Optional message")
