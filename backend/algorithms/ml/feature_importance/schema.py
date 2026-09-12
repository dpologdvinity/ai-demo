"""Pydantic schemas for Feature Importance Analysis API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class FeatureImportanceRequest(BaseModel):
    """Request schema for Feature Importance Analysis.

    Attributes:
        method: Importance method to use
        model_type: Type of model to train
        n_estimators: Number of trees/estimators in ensemble
        top_k: Number of top features to highlight
        dataset: Dataset to use for analysis
        random_state: Random seed for reproducibility
    """

    method: str = Field(
        default='tree',
        description="Importance method: 'tree', 'permutation', or 'all'"
    )
    model_type: str = Field(
        default='random_forest',
        description="Model type: 'random_forest', 'xgboost', or 'gradient_boosting'"
    )
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of trees/estimators in the ensemble"
    )
    top_k: int = Field(
        default=10,
        ge=5,
        le=30,
        description="Number of top features to show"
    )
    dataset: str = Field(
        default='housing',
        description="Dataset: 'housing', 'diabetes', or 'wine'"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "method": "all",
                "model_type": "random_forest",
                "n_estimators": 100,
                "top_k": 10,
                "dataset": "housing",
                "random_state": 42
            }
        }


class FeatureImportanceResponse(BaseModel):
    """Response schema for Feature Importance Analysis results.

    Attributes:
        metrics: Model performance metrics
        feature_importance: Dictionary of importance scores by method
        feature_rankings: Rankings of features by importance
        correlation_matrix: Feature correlation matrix data
        cumulative_importance: Cumulative importance curve data
        visualization_data: Data for various visualizations
        execution_time_ms: Total execution time in milliseconds
        model_info: Additional model and dataset information
        statistics: Statistical summary of importance scores
    """

    metrics: Dict[str, float] = Field(
        description="Model performance metrics (R2, MSE, MAE for regression or accuracy for classification)"
    )
    feature_importance: Dict[str, Dict[str, float]] = Field(
        description="Feature importance scores by method (tree, permutation, etc.)"
    )
    feature_rankings: Dict[str, List[Dict[str, Any]]] = Field(
        description="Rankings of features by importance for each method"
    )
    correlation_matrix: Dict[str, Any] = Field(
        description="Feature correlation matrix data"
    )
    cumulative_importance: List[Dict[str, Any]] = Field(
        description="Cumulative importance curve data"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Formatted data for visualizations"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    model_info: Dict[str, Any] = Field(
        description="Model and dataset metadata"
    )
    statistics: Dict[str, Any] = Field(
        description="Statistical summary of importance scores"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metrics": {
                    "r2_score": 0.812,
                    "mse": 0.456,
                    "mae": 0.523
                },
                "feature_importance": {
                    "tree": {
                        "feature1": 0.25,
                        "feature2": 0.18
                    },
                    "permutation": {
                        "feature1": 0.22,
                        "feature2": 0.19
                    }
                },
                "execution_time_ms": 1234.5
            }
        }
