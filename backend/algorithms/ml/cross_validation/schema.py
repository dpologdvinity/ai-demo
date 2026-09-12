"""
Pydantic schemas for Cross-Validation algorithm.

This module defines request and response models for the Cross-Validation algorithm,
including parameter validation and response structure.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class CrossValidationRequest(BaseModel):
    """Request model for Cross-Validation training.

    Attributes:
        cv_method: Cross-validation strategy to use
        n_splits: Number of folds/splits
        model_type: Type of model to evaluate
        scoring: Scoring metric to use
        shuffle: Whether to shuffle data before splitting
        dataset: Dataset to use for evaluation
        test_size: Size of test set (for shuffle_split)
        random_state: Random seed for reproducibility
    """

    cv_method: Literal['k_fold', 'stratified', 'shuffle_split', 'leave_one_out', 'time_series'] = Field(
        default='k_fold',
        description="Cross-validation strategy"
    )
    n_splits: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of folds/splits"
    )
    model_type: Literal['random_forest', 'logistic', 'svm', 'knn'] = Field(
        default='random_forest',
        description="Model to evaluate"
    )
    scoring: Literal['accuracy', 'f1', 'precision', 'recall', 'roc_auc'] = Field(
        default='accuracy',
        description="Scoring metric"
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle data before splitting"
    )
    dataset: Literal['iris', 'wine', 'breast_cancer'] = Field(
        default='iris',
        description="Dataset to use"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Test size for shuffle_split"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class FoldMetrics(BaseModel):
    """Metrics for a single fold.

    Attributes:
        fold_index: Index of the fold
        train_score: Training score for this fold
        test_score: Test score for this fold
        train_size: Number of samples in training set
        test_size: Number of samples in test set
        class_distribution: Distribution of classes in train/test sets
    """

    fold_index: int = Field(description="Fold index")
    train_score: float = Field(description="Training score")
    test_score: float = Field(description="Test score")
    train_size: int = Field(description="Training set size")
    test_size: int = Field(description="Test set size")
    class_distribution: Dict[str, Dict[str, int]] = Field(
        description="Class distribution in train and test sets"
    )


class CrossValidationMetrics(BaseModel):
    """Aggregate metrics from Cross-Validation.

    Attributes:
        mean_score: Mean test score across folds
        std_score: Standard deviation of test scores
        min_score: Minimum test score
        max_score: Maximum test score
        mean_train_score: Mean training score
        std_train_score: Standard deviation of training scores
        stability_coefficient: Coefficient of variation (std/mean)
        scoring_metric: The metric used for scoring
    """

    mean_score: float = Field(description="Mean test score")
    std_score: float = Field(description="Standard deviation of test scores")
    min_score: float = Field(description="Minimum test score")
    max_score: float = Field(description="Maximum test score")
    mean_train_score: float = Field(description="Mean training score")
    std_train_score: float = Field(description="Standard deviation of training scores")
    stability_coefficient: float = Field(description="Coefficient of variation (std/mean)")
    scoring_metric: str = Field(description="Scoring metric used")


class ConfusionMatrixData(BaseModel):
    """Confusion matrix for a fold.

    Attributes:
        fold_index: Index of the fold
        matrix: Confusion matrix
        class_labels: Class labels
    """

    fold_index: int = Field(description="Fold index")
    matrix: List[List[int]] = Field(description="Confusion matrix")
    class_labels: List[str] = Field(description="Class labels")


class VisualizationData(BaseModel):
    """Visualization data for Cross-Validation results.

    Attributes:
        fold_performance: Per-fold performance data
        score_distribution: Score distribution data
        train_test_splits: Visualization of train/test split layout
        confusion_matrices: Per-fold confusion matrices
        class_distributions: Class distribution per fold
    """

    fold_performance: List[Dict[str, Any]] = Field(
        description="Performance data for each fold"
    )
    score_distribution: List[Dict[str, Any]] = Field(
        description="Score distribution across folds"
    )
    train_test_splits: List[Dict[str, Any]] = Field(
        description="Train/test split visualization data"
    )
    confusion_matrices: List[ConfusionMatrixData] = Field(
        description="Confusion matrices for each fold"
    )
    class_distributions: List[Dict[str, Any]] = Field(
        description="Class distribution per fold"
    )


class CrossValidationResponse(BaseModel):
    """Response model for Cross-Validation training results.

    Attributes:
        success: Whether the cross-validation was successful
        metrics: Aggregate performance metrics
        fold_metrics: Per-fold metrics
        visualization_data: Data for frontend visualization
        execution_time_ms: Time taken for cross-validation in milliseconds
        parameters: The parameters used for this run
        message: Optional message about the cross-validation
    """

    success: bool = Field(description="Whether cross-validation succeeded")
    metrics: CrossValidationMetrics = Field(description="Aggregate metrics")
    fold_metrics: List[FoldMetrics] = Field(description="Per-fold metrics")
    visualization_data: VisualizationData = Field(description="Visualization data")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    parameters: Dict[str, Any] = Field(description="Parameters used")
    message: Optional[str] = Field(default=None, description="Optional message")
