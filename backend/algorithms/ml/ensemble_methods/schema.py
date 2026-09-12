"""
Pydantic schemas for Ensemble Methods algorithm.

This module defines request and response models for ensemble learning techniques
including Bagging, Boosting, Stacking, and Voting methods.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class EnsembleMethodsRequest(BaseModel):
    """Request model for Ensemble Methods training and prediction.

    Attributes:
        method: Ensemble method to use (bagging, boosting, stacking, voting, all)
        n_estimators: Number of base models in the ensemble
        base_model: Type of base estimator to use
        max_samples: Ratio of samples to draw for bagging
        learning_rate: Learning rate for boosting algorithms
        test_size: Proportion of dataset to use for testing
        random_state: Random seed for reproducibility
        dataset_name: Name of dataset to use
        normalize: Whether to normalize features
    """

    method: Literal['bagging', 'boosting', 'stacking', 'voting', 'all'] = Field(
        default='voting',
        description="Ensemble method to use"
    )
    n_estimators: int = Field(
        default=10,
        ge=3,
        le=100,
        description="Number of base models in the ensemble"
    )
    base_model: Literal['decision_tree', 'svm', 'knn', 'logistic'] = Field(
        default='decision_tree',
        description="Base estimator type"
    )
    max_samples: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Ratio of samples to draw for bagging (0.5-1.0)"
    )
    learning_rate: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Learning rate for boosting algorithms"
    )
    test_size: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="Proportion of data to use for testing"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducible results"
    )
    dataset_name: str = Field(
        default='wine',
        description="Dataset to use (wine, breast_cancer, iris)"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize features before training"
    )


class EnsembleMetrics(BaseModel):
    """Classification metrics from ensemble model.

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


class SingleModelMetrics(BaseModel):
    """Performance metrics for a single base model."""

    model_name: str = Field(description="Name of the model")
    accuracy: float = Field(description="Model accuracy")
    precision: float = Field(description="Model precision")
    recall: float = Field(description="Model recall")
    f1_score: float = Field(description="Model F1 score")


class PerformanceComparisonData(BaseModel):
    """Performance comparison between single models and ensemble."""

    single_models: List[SingleModelMetrics] = Field(
        description="Individual model performances"
    )
    ensemble_metrics: Dict[str, float] = Field(
        description="Ensemble performance metrics"
    )


class DiversityMetrics(BaseModel):
    """Diversity metrics for the ensemble."""

    disagreement: float = Field(
        description="Average disagreement between models (0-1)"
    )
    avg_correlation: float = Field(
        description="Average correlation between model predictions (-1 to 1)"
    )
    q_statistic: float = Field(
        description="Q-statistic measuring pairwise diversity (-1 to 1)"
    )


class VotingData(BaseModel):
    """Voting pattern data for ensemble predictions."""

    sample_indices: List[int] = Field(description="Sample indices")
    model_predictions: List[List[int]] = Field(
        description="Predictions from each model for each sample"
    )
    ensemble_predictions: List[int] = Field(
        description="Final ensemble predictions"
    )
    actual_labels: List[int] = Field(description="True labels")
    agreement_scores: List[float] = Field(
        description="Agreement score for each sample (0-1)"
    )


class VisualizationData(BaseModel):
    """Visualization data for Ensemble Methods results.

    Attributes:
        performance_comparison: Comparison of single vs ensemble performance
        feature_importance: Feature importance from the ensemble
        confusion_matrix: Confusion matrix for classification results
        class_labels: Names of the target classes
        diversity_metrics: Diversity metrics between base models
        voting_patterns: Voting patterns and agreement data
        individual_predictions: Predictions from each base model
    """

    performance_comparison: PerformanceComparisonData = Field(
        description="Performance comparison data"
    )
    feature_importance: Dict[str, float] = Field(
        description="Feature importance scores from ensemble"
    )
    confusion_matrix: List[List[int]] = Field(
        description="Confusion matrix showing predictions vs actual"
    )
    class_labels: List[str] = Field(
        description="Names of the classification classes"
    )
    diversity_metrics: DiversityMetrics = Field(
        description="Ensemble diversity metrics"
    )
    voting_patterns: VotingData = Field(
        description="Voting patterns and model agreement data"
    )
    individual_predictions: Dict[str, List[int]] = Field(
        description="Predictions from each individual model"
    )


class EnsembleMethodsResponse(BaseModel):
    """Response model for Ensemble Methods training results.

    Attributes:
        success: Whether the training was successful
        metrics: Ensemble performance metrics
        predictions: List of predicted class labels for test set
        actual: List of actual class labels for test set
        visualization_data: Data for frontend visualization
        execution_time_ms: Time taken for training and prediction in milliseconds
        parameters: The parameters used for this training run
        model_info: Information about the trained ensemble
        message: Optional message about the training
    """

    success: bool = Field(description="Whether training succeeded")
    metrics: EnsembleMetrics = Field(description="Performance metrics")
    predictions: List[int] = Field(description="Predicted labels")
    actual: List[int] = Field(description="Actual labels")
    visualization_data: VisualizationData = Field(description="Visualization data")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    parameters: Dict[str, Any] = Field(description="Parameters used")
    model_info: Dict[str, Any] = Field(description="Model information")
    message: Optional[str] = Field(default=None, description="Optional message")
