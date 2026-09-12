"""
Pydantic schemas for Decision Tree Classifier API endpoints.

This module defines the request and response models for the Decision Tree
classifier API endpoints, ensuring type safety and validation.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class DecisionTreeRequest(BaseModel):
    """Request schema for Decision Tree training.

    Attributes:
        max_depth: Maximum depth of the tree. None means unlimited.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        criterion: Split quality measure ('gini' or 'entropy').
        dataset_name: Name of dataset to use (default: 'iris').
        random_state: Random seed for reproducibility.

    Example:
        >>> request = DecisionTreeRequest(
        ...     max_depth=5,
        ...     min_samples_split=2,
        ...     min_samples_leaf=1,
        ...     criterion='gini'
        ... )
    """

    max_depth: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum tree depth (1-20, or None for unlimited)"
    )
    min_samples_split: int = Field(
        default=2,
        ge=2,
        le=20,
        description="Minimum samples to split an internal node (2-20)"
    )
    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum samples per leaf node (1-10)"
    )
    criterion: str = Field(
        default='gini',
        pattern='^(gini|entropy)$',
        description="Split quality measure ('gini' or 'entropy')"
    )
    dataset_name: str = Field(
        default='iris',
        description="Dataset to use for training"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class DecisionTreeResponse(BaseModel):
    """Response schema for Decision Tree training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including accuracy, precision, recall, F1
        predictions: Model predictions on test set
        visualization_data: Tree structure and confusion matrix data
        execution_time_ms: Training time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed

    Example:
        >>> response = DecisionTreeResponse(
        ...     success=True,
        ...     metrics={'accuracy': 0.95, 'f1_score': 0.94},
        ...     predictions={'y_test': [...], 'y_pred': [...]},
        ...     visualization_data={...},
        ...     execution_time_ms=15.3,
        ...     parameters_used={'max_depth': 5}
        ... )
    """

    success: bool
    metrics: Dict[str, float]
    predictions: Optional[Dict[str, List[Any]]] = None
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any]
    error: Optional[str] = None


class DecisionTreeInfoResponse(BaseModel):
    """Response schema for Decision Tree algorithm information.

    Attributes:
        metadata: Complete algorithm metadata
        available_datasets: List of compatible dataset names

    Example:
        >>> info = DecisionTreeInfoResponse(
        ...     metadata={...},
        ...     available_datasets=['iris', 'wine', 'digits']
        ... )
    """

    metadata: Dict[str, Any]
    available_datasets: List[str]
