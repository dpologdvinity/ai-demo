"""Request and response schemas for Anomaly Detection algorithm."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class AnomalyDetectionParameters(BaseModel):
    """Parameters for Anomaly Detection algorithm.

    Attributes:
        method: Detection method to use
        contamination: Expected anomaly ratio in the dataset
        n_estimators: Number of trees for Isolation Forest
        kernel: Kernel type for One-Class SVM
        n_neighbors: Number of neighbors for LOF
        dataset: Dataset type to use
        random_state: Random seed for reproducibility
        n_samples: Number of samples to generate
    """

    method: str = Field(
        default='isolation_forest',
        description="Detection method: 'isolation_forest', 'one_class_svm', 'lof', or 'compare'"
    )
    contamination: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Expected anomaly ratio in the dataset"
    )
    n_estimators: int = Field(
        default=100,
        ge=50,
        le=500,
        description="Number of trees for Isolation Forest"
    )
    kernel: str = Field(
        default='rbf',
        description="Kernel type for One-Class SVM: 'rbf', 'linear', 'poly', 'sigmoid'"
    )
    n_neighbors: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of neighbors for LOF"
    )
    dataset: str = Field(
        default='synthetic',
        description="Dataset type: 'synthetic', 'fraud', 'network'"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    n_samples: int = Field(
        default=300,
        ge=100,
        le=1000,
        description="Number of samples to generate"
    )

    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate that method is supported."""
        valid_methods = ['isolation_forest', 'one_class_svm', 'lof', 'compare']
        if v not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        return v

    @field_validator('kernel')
    @classmethod
    def validate_kernel(cls, v: str) -> str:
        """Validate that kernel is supported."""
        valid_kernels = ['rbf', 'linear', 'poly', 'sigmoid']
        if v not in valid_kernels:
            raise ValueError(f"kernel must be one of {valid_kernels}")
        return v

    @field_validator('dataset')
    @classmethod
    def validate_dataset(cls, v: str) -> str:
        """Validate that dataset type is supported."""
        valid_datasets = ['synthetic', 'fraud', 'network']
        if v not in valid_datasets:
            raise ValueError(f"dataset must be one of {valid_datasets}")
        return v


class AnomalyInfo(BaseModel):
    """Information about anomaly detection results.

    Attributes:
        n_anomalies: Number of points classified as anomalies
        n_normal: Number of points classified as normal
        anomaly_ratio: Proportion of anomalies detected
        method_used: Detection method that was used
    """

    n_anomalies: int
    n_normal: int
    anomaly_ratio: float
    method_used: str


class MethodComparison(BaseModel):
    """Comparison of different anomaly detection methods.

    Attributes:
        method_name: Name of the method
        n_anomalies: Number of anomalies detected
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        execution_time_ms: Execution time in milliseconds
    """

    method_name: str
    n_anomalies: int
    precision: float
    recall: float
    f1_score: float
    execution_time_ms: float


class AnomalyDetectionResponse(BaseModel):
    """Response schema for Anomaly Detection training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics
        anomaly_info: Information about detected anomalies
        visualization_data: Data formatted for visualization
        execution_time_ms: Training execution time in milliseconds
        parameters_used: Actual parameters used for training
        method_comparison: Comparison of methods (if compare mode used)
        error: Error message if training failed
    """

    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    anomaly_info: Optional[AnomalyInfo] = None
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    method_comparison: Optional[List[MethodComparison]] = None
    error: Optional[str] = None
