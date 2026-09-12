"""Request and response schemas for DBSCAN Clustering algorithm."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class DBSCANParameters(BaseModel):
    """Parameters for DBSCAN clustering algorithm.

    Attributes:
        eps: Maximum distance between two samples for one to be considered
            in the neighborhood of the other (default: 0.5)
        min_samples: Minimum number of samples in a neighborhood for a point
            to be considered a core point (default: 5)
        metric: Distance metric to use (default: 'euclidean')
        random_state: Random seed for reproducibility (default: 42)
        n_samples: Number of data points to generate (default: 300)
        noise: Standard deviation of Gaussian noise added to data (default: 0.1)
    """

    eps: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Maximum distance between samples in neighborhood (epsilon)"
    )
    min_samples: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Minimum samples in neighborhood to form core point"
    )
    metric: str = Field(
        default='euclidean',
        description="Distance metric ('euclidean' or 'manhattan')"
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
    noise: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Noise level for moons dataset generation"
    )

    @field_validator('eps')
    @classmethod
    def validate_eps(cls, v: float) -> float:
        """Validate that eps is within valid range."""
        if not 0.1 <= v <= 2.0:
            raise ValueError("eps must be between 0.1 and 2.0")
        return v

    @field_validator('min_samples')
    @classmethod
    def validate_min_samples(cls, v: int) -> int:
        """Validate that min_samples is within valid range."""
        if not 2 <= v <= 20:
            raise ValueError("min_samples must be between 2 and 20")
        return v

    @field_validator('metric')
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate that metric is supported."""
        if v not in ['euclidean', 'manhattan']:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        return v


class ClusterInfo(BaseModel):
    """Information about a single cluster.

    Attributes:
        cluster_id: Unique identifier for the cluster (-1 for noise points)
        size: Number of points in the cluster
        is_noise: Whether this represents the noise cluster
    """

    cluster_id: int
    size: int
    is_noise: bool = False


class DBSCANResponse(BaseModel):
    """Response schema for DBSCAN training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including n_clusters, n_noise, silhouette score
        clusters: Information about each cluster including noise
        visualization_data: Data formatted for scatter plot visualization
        execution_time_ms: Training execution time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed
    """

    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    clusters: List[ClusterInfo] = Field(default_factory=list)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
