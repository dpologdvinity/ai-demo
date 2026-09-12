"""Pydantic schemas for Random Forest API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RandomForestRequest(BaseModel):
    """Request schema for Random Forest training.

    Attributes:
        n_estimators: Number of trees in the forest (default: 100, range: 10-500)
        max_depth: Maximum depth of each tree (default: None for unlimited, range: 1-30)
        min_samples_split: Minimum samples required to split an internal node (default: 2, range: 2-20)
        max_features: Number of features to consider when looking for the best split
            Options: 'sqrt', 'log2', or None (all features)
        random_state: Random seed for reproducibility (default: 42)
    """

    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of trees in the forest"
    )
    max_depth: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="Maximum depth of the tree. None means unlimited."
    )
    min_samples_split: int = Field(
        default=2,
        ge=2,
        le=20,
        description="Minimum number of samples required to split an internal node"
    )
    max_features: str = Field(
        default='sqrt',
        description="Number of features to consider: 'sqrt', 'log2', or 'None'"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "max_features": "sqrt",
                "random_state": 42
            }
        }


class RandomForestResponse(BaseModel):
    """Response schema for Random Forest training results.

    Attributes:
        metrics: Dictionary of performance metrics (accuracy, precision, recall, f1_score)
        predictions: List of predicted class labels for test set
        feature_importance: Dictionary mapping feature names to importance scores
        confusion_matrix: 2D list representing the confusion matrix
        execution_time_ms: Training and prediction time in milliseconds
        model_info: Additional model information (n_estimators, max_depth, etc.)
    """

    metrics: Dict[str, float] = Field(
        description="Performance metrics including accuracy, precision, recall, f1_score"
    )
    predictions: List[int] = Field(
        description="Predicted class labels for the test set"
    )
    feature_importance: Dict[str, float] = Field(
        description="Feature importance scores for each feature"
    )
    confusion_matrix: List[List[int]] = Field(
        description="Confusion matrix as a 2D list"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    model_info: Dict[str, Any] = Field(
        description="Additional model information and parameters"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metrics": {
                    "accuracy": 0.963,
                    "precision": 0.965,
                    "recall": 0.960,
                    "f1_score": 0.962
                },
                "predictions": [0, 1, 2, 1, 0],
                "feature_importance": {
                    "alcohol": 0.125,
                    "malic_acid": 0.082,
                    "ash": 0.015
                },
                "confusion_matrix": [[19, 0, 0], [0, 20, 1], [0, 1, 13]],
                "execution_time_ms": 45.2,
                "model_info": {
                    "n_estimators": 100,
                    "max_depth": None,
                    "total_trees": 100
                }
            }
        }
