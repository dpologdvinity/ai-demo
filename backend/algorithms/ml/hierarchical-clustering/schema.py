"""Pydantic schemas for Hierarchical Clustering API requests and responses."""

from typing import Dict, List, Any
from pydantic import BaseModel, Field


class HierarchicalClusteringRequest(BaseModel):
    """Request schema for training Hierarchical Clustering model.

    Attributes:
        n_clusters: Number of clusters to find (default: 3, min: 2, max: 10)
        linkage: Linkage criterion (default: 'ward')
        affinity: Distance metric (default: 'euclidean')
        n_samples: Number of samples to generate for blob dataset (default: 300)
    """

    n_clusters: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of clusters to find"
    )
    linkage: str = Field(
        default="ward",
        description="Linkage criterion: 'ward', 'complete', 'average', or 'single'"
    )
    affinity: str = Field(
        default="euclidean",
        description="Distance metric: 'euclidean', 'manhattan', or 'cosine'"
    )
    n_samples: int = Field(
        default=300,
        ge=50,
        le=1000,
        description="Number of samples to generate"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "n_clusters": 3,
                "linkage": "ward",
                "affinity": "euclidean",
                "n_samples": 300
            }
        }


class HierarchicalClusteringResponse(BaseModel):
    """Response schema for Hierarchical Clustering training results.

    Attributes:
        metrics: Dictionary containing clustering evaluation metrics
        predictions: Cluster assignments for each sample
        visualization_data: Data for visualizing the clustering results
        execution_time_ms: Time taken to train the model in milliseconds
        model_info: Additional information about the trained model
    """

    metrics: Dict[str, float] = Field(
        description="Clustering evaluation metrics (silhouette, davies_bouldin, calinski_harabasz)"
    )
    predictions: List[int] = Field(
        description="Cluster label for each sample"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Data for dendrogram and scatter plot visualization"
    )
    execution_time_ms: float = Field(
        description="Training time in milliseconds"
    )
    model_info: Dict[str, Any] = Field(
        description="Model configuration and learned parameters"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metrics": {
                    "silhouette_score": 0.65,
                    "davies_bouldin_score": 0.45,
                    "calinski_harabasz_score": 350.2
                },
                "predictions": [0, 0, 1, 1, 2, 2],
                "visualization_data": {
                    "scatter": {
                        "X": [[1.0, 2.0], [1.1, 2.1]],
                        "labels": [0, 0]
                    },
                    "dendrogram": {
                        "linkage_matrix": [[0, 1, 0.5, 2]],
                        "n_clusters": 3
                    }
                },
                "execution_time_ms": 45.2,
                "model_info": {
                    "n_clusters": 3,
                    "linkage": "ward",
                    "affinity": "euclidean",
                    "n_samples": 300
                }
            }
        }
