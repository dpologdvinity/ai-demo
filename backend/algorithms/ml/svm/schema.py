"""Pydantic schemas for SVM API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SVMRequest(BaseModel):
    """Request model for SVM training.

    Attributes:
        C: Regularization parameter (default: 1.0)
        kernel: Kernel type - 'linear', 'poly', 'rbf', or 'sigmoid' (default: 'rbf')
        gamma: Kernel coefficient - 'scale' or 'auto' (default: 'scale')
        degree: Polynomial degree for poly kernel (default: 3)
    """

    C: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Regularization parameter. Smaller values specify stronger regularization."
    )
    kernel: str = Field(
        default='rbf',
        description="Kernel type: 'linear', 'poly', 'rbf', or 'sigmoid'"
    )
    gamma: str = Field(
        default='scale',
        description="Kernel coefficient: 'scale' or 'auto'"
    )
    degree: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Degree of polynomial kernel (ignored by other kernels)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "degree": 3
            }
        }


class SVMMetrics(BaseModel):
    """Model performance metrics.

    Attributes:
        accuracy: Overall accuracy score (0-1)
        precision: Macro-averaged precision score
        recall: Macro-averaged recall score
        f1_score: Macro-averaged F1 score
        confusion_matrix: Confusion matrix as nested list
    """

    accuracy: float = Field(description="Overall accuracy (0-1)")
    precision: float = Field(description="Macro-averaged precision")
    recall: float = Field(description="Macro-averaged recall")
    f1_score: float = Field(description="Macro-averaged F1 score")
    confusion_matrix: List[List[int]] = Field(description="Confusion matrix")


class SVMModelInfo(BaseModel):
    """Trained model information.

    Attributes:
        C: Regularization parameter used
        kernel: Kernel type used
        gamma: Gamma value used
        degree: Polynomial degree used
        n_support_vectors: Total number of support vectors
        support_vectors_per_class: Number of support vectors per class
        n_features: Number of input features
        n_classes: Number of output classes
        classes: List of class labels
    """

    C: float
    kernel: str
    gamma: Any
    degree: int
    n_support_vectors: int
    support_vectors_per_class: List[int]
    n_features: int
    n_classes: int
    classes: List[int]


class VisualizationData(BaseModel):
    """Data for visualizing SVM results.

    Attributes:
        training_data: Training data points with predictions
        test_data: Test data points with predictions
        support_vectors: Support vector coordinates
        decision_boundary: Decision boundary mesh data (optional)
        feature_names: Names of features used
        target_names: Names of target classes
    """

    training_data: List[Dict[str, Any]] = Field(
        description="Training data points with features and labels"
    )
    test_data: List[Dict[str, Any]] = Field(
        description="Test data points with features and predictions"
    )
    support_vectors: List[Dict[str, Any]] = Field(
        description="Support vector coordinates"
    )
    decision_boundary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Decision boundary mesh data for visualization"
    )
    feature_names: List[str] = Field(description="Feature names")
    target_names: List[str] = Field(description="Target class names")


class SVMResponse(BaseModel):
    """Response model for SVM training results.

    Attributes:
        metrics: Model performance metrics
        model_info: Trained model information
        visualization_data: Data for visualization
        execution_time_ms: Total execution time in milliseconds
    """

    metrics: SVMMetrics = Field(description="Performance metrics")
    model_info: SVMModelInfo = Field(description="Model configuration and parameters")
    visualization_data: VisualizationData = Field(description="Visualization data")
    execution_time_ms: float = Field(description="Total execution time in milliseconds")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metrics": {
                    "accuracy": 0.96,
                    "precision": 0.95,
                    "recall": 0.96,
                    "f1_score": 0.95,
                    "confusion_matrix": [[15, 0, 0], [0, 13, 1], [0, 0, 16]]
                },
                "model_info": {
                    "C": 1.0,
                    "kernel": "rbf",
                    "gamma": "scale",
                    "degree": 3,
                    "n_support_vectors": 23,
                    "support_vectors_per_class": [8, 7, 8],
                    "n_features": 4,
                    "n_classes": 3,
                    "classes": [0, 1, 2]
                },
                "visualization_data": {
                    "training_data": [],
                    "test_data": [],
                    "support_vectors": [],
                    "feature_names": ["sepal length", "sepal width"],
                    "target_names": ["setosa", "versicolor", "virginica"]
                },
                "execution_time_ms": 15.5
            }
        }
