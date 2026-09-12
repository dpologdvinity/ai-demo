"""Request and response schemas for K-Means Clustering algorithm."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class KMeansParameters(BaseModel):
    """Parameters for K-Means clustering algorithm.

    Attributes:
        n_clusters: Number of clusters K to form (default: 3)
        max_iter: Maximum number of iterations for convergence (default: 300)
        n_init: Number of times the algorithm runs with different centroid seeds (default: 10)
        random_state: Random seed for reproducibility (default: 42)
        n_samples: Number of data points to generate (default: 300)
    """

    n_clusters: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of clusters K to form"
    )
    max_iter: int = Field(
        default=300,
        ge=50,
        le=1000,
        description="Maximum number of iterations"
    )
    n_init: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of initializations with different centroid seeds"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    n_samples: int = Field(
        default=300,
        ge=100,
        le=1000,
        description="Number of data points to generate"
    )

    @field_validator('n_clusters')
    @classmethod
    def validate_n_clusters(cls, v: int) -> int:
        """Validate that n_clusters is within valid range."""
        if not 2 <= v <= 10:
            raise ValueError("n_clusters must be between 2 and 10")
        return v

    @field_validator('max_iter')
    @classmethod
    def validate_max_iter(cls, v: int) -> int:
        """Validate that max_iter is within valid range."""
        if not 50 <= v <= 1000:
            raise ValueError("max_iter must be between 50 and 1000")
        return v


class ClusterInfo(BaseModel):
    """Information about a single cluster.

    Attributes:
        cluster_id: Unique identifier for the cluster
        center: Coordinates of the cluster centroid
        size: Number of points in the cluster
    """

    cluster_id: int
    center: List[float]
    size: int


class KMeansResponse(BaseModel):
    """Response schema for K-Means training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including inertia and silhouette score
        clusters: Information about each cluster
        visualization_data: Data formatted for scatter plot visualization
        execution_time_ms: Training execution time in milliseconds
        parameters_used: Actual parameters used for training
        n_iterations: Number of iterations until convergence
        error: Error message if training failed
    """

    success: bool
    metrics: Dict[str, float] = Field(default_factory=dict)
    clusters: List[ClusterInfo] = Field(default_factory=list)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    n_iterations: Optional[int] = None
    error: Optional[str] = None
