"""Request and response schemas for Isolation Forest algorithm."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class IsolationForestParameters(BaseModel):
    """Parameters for Isolation Forest anomaly detection algorithm.

    Attributes:
        n_estimators: Number of isolation trees in the forest (default: 100)
        contamination: Expected proportion of outliers in dataset (default: 0.1)
        max_samples: Number of samples to draw to train each tree (default: 'auto')
        random_state: Random seed for reproducibility (default: 42)
        n_samples: Total number of data points to generate (default: 300)
        n_outliers_ratio: Ratio of outliers to inject into the dataset (default: 0.1)
    """

    n_estimators: int = Field(
        default=100,
        ge=50,
        le=300,
        description="Number of isolation trees in the forest"
    )
    contamination: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Expected proportion of outliers in the dataset"
    )
    max_samples: str | int = Field(
        default='auto',
        description="Number of samples to draw to train each tree ('auto' or integer)"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    n_samples: int = Field(
        default=300,
        ge=100,
        le=1000,
        description="Total number of data points to generate"
    )
    n_outliers_ratio: float = Field(
        default=0.1,
        ge=0.01,
        le=0.3,
        description="Ratio of outliers to inject into dataset"
    )

    @field_validator('n_estimators')
    @classmethod
    def validate_n_estimators(cls, v: int) -> int:
        """Validate that n_estimators is within valid range."""
        if not 50 <= v <= 300:
            raise ValueError("n_estimators must be between 50 and 300")
        return v

    @field_validator('contamination')
    @classmethod
    def validate_contamination(cls, v: float) -> float:
        """Validate that contamination is within valid range."""
        if not 0.01 <= v <= 0.5:
            raise ValueError("contamination must be between 0.01 and 0.5")
        return v

    @field_validator('max_samples')
    @classmethod
    def validate_max_samples(cls, v: str | int) -> str | int:
        """Validate that max_samples is valid."""
        if isinstance(v, str) and v != 'auto':
            raise ValueError("max_samples must be 'auto' or an integer")
        if isinstance(v, int) and (v < 100 or v > 1000):
            raise ValueError("max_samples must be between 100 and 1000 when specified as integer")
        return v


class AnomalyInfo(BaseModel):
    """Information about anomaly detection results.

    Attributes:
        n_anomalies: Number of points classified as anomalies
        n_normal: Number of points classified as normal
        anomaly_ratio: Proportion of anomalies detected
    """

    n_anomalies: int
    n_normal: int
    anomaly_ratio: float


class IsolationForestResponse(BaseModel):
    """Response schema for Isolation Forest training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including anomaly counts and scores
        anomaly_info: Information about detected anomalies
        visualization_data: Data formatted for scatter plot visualization
        execution_time_ms: Training execution time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed
    """

    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    anomaly_info: Optional[AnomalyInfo] = None
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
