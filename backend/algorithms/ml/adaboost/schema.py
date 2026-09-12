"""
Pydantic schemas for AdaBoost (Adaptive Boosting) algorithm.

This module defines request and response models for the AdaBoost algorithm,
including parameter validation and response structure.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class AdaBoostRequest(BaseModel):
    """Request model for AdaBoost training and prediction.

    Attributes:
        n_estimators: Number of weak learners (boosting rounds)
        learning_rate: Weight shrinkage applied to each classifier
        algorithm: Boosting algorithm variant to use
        test_size: Proportion of dataset to use for testing
        random_state: Random seed for reproducibility
    """

    n_estimators: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of weak learners to train sequentially"
    )
    learning_rate: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Weight applied to each classifier (shrinkage parameter)"
    )
    algorithm: Literal['SAMME', 'SAMME.R'] = Field(
        default='SAMME.R',
        description="Boosting algorithm: 'SAMME' (discrete) or 'SAMME.R' (real, uses probability estimates)"
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


class AdaBoostMetrics(BaseModel):
    """Classification metrics from AdaBoost model.

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


class LearningCurveData(BaseModel):
    """Learning curve data showing performance over boosting rounds.

    Attributes:
        estimators: Number of estimators at each evaluation point
        train_scores: Training accuracy at each point
        test_scores: Test accuracy at each point
        estimator_errors: Training error for each estimator
        estimator_weights: Weight assigned to each estimator
    """

    estimators: List[int] = Field(description="Number of estimators")
    train_scores: List[float] = Field(description="Training scores")
    test_scores: List[float] = Field(description="Test scores")
    estimator_errors: List[float] = Field(description="Error for each weak learner")
    estimator_weights: List[float] = Field(description="Weight for each weak learner")


class VisualizationData(BaseModel):
    """Visualization data for AdaBoost results.

    Attributes:
        learning_curve: Learning curve showing accuracy vs number of estimators
        feature_importance: Feature importance scores
        confusion_matrix: Confusion matrix for classification results
        class_labels: Names of the target classes
    """

    learning_curve: LearningCurveData = Field(
        description="Learning curve data across boosting rounds"
    )
    feature_importance: Dict[str, float] = Field(
        description="Feature importance scores from the ensemble"
    )
    confusion_matrix: List[List[int]] = Field(
        description="Confusion matrix showing prediction vs actual"
    )
    class_labels: List[str] = Field(
        description="Names of the classification classes"
    )


class AdaBoostResponse(BaseModel):
    """Response model for AdaBoost training results.

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
    metrics: AdaBoostMetrics = Field(description="Performance metrics")
    predictions: List[int] = Field(description="Predicted labels")
    actual: List[int] = Field(description="Actual labels")
    visualization_data: VisualizationData = Field(description="Visualization data")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    parameters: Dict[str, Any] = Field(description="Parameters used")
    message: Optional[str] = Field(default=None, description="Optional message")
