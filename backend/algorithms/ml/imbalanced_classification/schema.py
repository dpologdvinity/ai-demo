"""Pydantic schemas for Imbalanced Classification.

This module defines request and response models for handling imbalanced
classification tasks using various sampling strategies and class weights.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class ImbalancedClassificationRequest(BaseModel):
    """Request model for Imbalanced Classification training.

    Attributes:
        strategy: Resampling strategy to use
        target_ratio: Target ratio for minority class (0.1 to 1.0)
        k_neighbors: Number of neighbors for SMOTE
        sampling_strategy: Sampling approach for resampling
        model_type: Type of classifier to use
        imbalance_ratio: Ratio of majority to minority class (10:1 or 100:1)
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    """

    strategy: Literal['smote', 'undersample', 'oversample', 'class_weights', 'compare'] = Field(
        default='smote',
        description="Balance strategy: SMOTE, undersampling, oversampling, class weights, or compare all"
    )
    target_ratio: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Target ratio for minority class after resampling"
    )
    k_neighbors: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Number of nearest neighbors for SMOTE algorithm"
    )
    sampling_strategy: Literal['auto', 'all', 'not_majority'] = Field(
        default='auto',
        description="Sampling strategy: auto (balance to target_ratio), all, or not_majority"
    )
    model_type: Literal['random_forest', 'logistic', 'xgboost'] = Field(
        default='random_forest',
        description="Type of classifier to use for evaluation"
    )
    imbalance_ratio: int = Field(
        default=10,
        ge=5,
        le=100,
        description="Ratio of majority to minority class (e.g., 10 means 10:1)"
    )
    test_size: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="Proportion of data to use for testing"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class StrategyMetrics(BaseModel):
    """Metrics for a single resampling strategy.

    Attributes:
        accuracy: Overall accuracy
        precision: Precision score (weighted or binary)
        recall: Recall score (weighted or binary)
        f1_score: F1 score (weighted or binary)
        roc_auc: ROC-AUC score
        pr_auc: Precision-Recall AUC score
        confusion_matrix: Confusion matrix as 2D list
        minority_recall: Recall for minority class
        minority_precision: Precision for minority class
    """

    accuracy: float = Field(description="Overall accuracy")
    precision: float = Field(description="Precision score")
    recall: float = Field(description="Recall score")
    f1_score: float = Field(description="F1 score")
    roc_auc: float = Field(description="ROC-AUC score")
    pr_auc: float = Field(description="Precision-Recall AUC score")
    confusion_matrix: List[List[int]] = Field(description="Confusion matrix")
    minority_recall: float = Field(description="Recall for minority class")
    minority_precision: float = Field(description="Precision for minority class")


class ImbalancedClassificationMetrics(BaseModel):
    """Overall metrics for imbalanced classification.

    Attributes:
        original: Metrics on original imbalanced data (baseline)
        smote: Metrics after SMOTE resampling
        undersample: Metrics after random undersampling
        oversample: Metrics after random oversampling
        class_weights: Metrics using class weight balancing
    """

    original: StrategyMetrics = Field(description="Baseline metrics on imbalanced data")
    smote: Optional[StrategyMetrics] = Field(default=None, description="SMOTE metrics")
    undersample: Optional[StrategyMetrics] = Field(default=None, description="Undersampling metrics")
    oversample: Optional[StrategyMetrics] = Field(default=None, description="Oversampling metrics")
    class_weights: Optional[StrategyMetrics] = Field(default=None, description="Class weights metrics")


class ClassDistribution(BaseModel):
    """Class distribution data.

    Attributes:
        class_0: Count for class 0 (majority)
        class_1: Count for class 1 (minority)
        ratio: Imbalance ratio
    """

    class_0: int = Field(description="Count for class 0")
    class_1: int = Field(description="Count for class 1")
    ratio: float = Field(description="Imbalance ratio (majority/minority)")


class ROCCurveData(BaseModel):
    """ROC curve data for a single strategy.

    Attributes:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Decision thresholds
        auc: Area under the curve
    """

    fpr: List[float] = Field(description="False positive rates")
    tpr: List[float] = Field(description="True positive rates")
    thresholds: List[float] = Field(description="Decision thresholds")
    auc: float = Field(description="Area under the ROC curve")


class PRCurveData(BaseModel):
    """Precision-Recall curve data for a single strategy.

    Attributes:
        precision: Precision values
        recall: Recall values
        thresholds: Decision thresholds
        auc: Area under the curve
    """

    precision: List[float] = Field(description="Precision values")
    recall: List[float] = Field(description="Recall values")
    thresholds: List[float] = Field(description="Decision thresholds")
    auc: float = Field(description="Area under the PR curve")


class VisualizationData(BaseModel):
    """Visualization data for imbalanced classification results.

    Attributes:
        original_distribution: Class distribution before resampling
        resampled_distribution: Class distribution after resampling
        roc_curves: ROC curves for each strategy
        pr_curves: Precision-Recall curves for each strategy
        metrics_comparison: Comparison of metrics across strategies
        sample_counts: Sample counts before/after resampling
    """

    original_distribution: ClassDistribution = Field(description="Original class distribution")
    resampled_distribution: Optional[ClassDistribution] = Field(
        default=None,
        description="Class distribution after resampling"
    )
    roc_curves: Dict[str, ROCCurveData] = Field(description="ROC curves for each strategy")
    pr_curves: Dict[str, PRCurveData] = Field(description="PR curves for each strategy")
    metrics_comparison: Dict[str, Dict[str, float]] = Field(
        description="Metrics comparison across strategies"
    )
    sample_counts: Dict[str, Dict[str, int]] = Field(
        description="Sample counts for each strategy"
    )


class ImbalancedClassificationResponse(BaseModel):
    """Response model for Imbalanced Classification training.

    Attributes:
        success: Whether training was successful
        metrics: Performance metrics for all strategies
        predictions: Predictions on test set
        visualization_data: Data for frontend visualization
        execution_time_ms: Execution time in milliseconds
        parameters_used: Parameters used for training
        message: Optional message
    """

    success: bool = Field(description="Whether training succeeded")
    metrics: ImbalancedClassificationMetrics = Field(description="Performance metrics")
    predictions: Dict[str, List[int]] = Field(description="Predictions for each strategy")
    visualization_data: VisualizationData = Field(description="Visualization data")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    parameters_used: Dict[str, Any] = Field(description="Parameters used")
    message: Optional[str] = Field(default=None, description="Optional message")
