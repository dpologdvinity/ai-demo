"""t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm implementation.

This package provides t-SNE dimensionality reduction for visualization of
high-dimensional data.
"""

from .model import TSNEModel
from .schema import TSNERequest, TSNEResponse

__all__ = ["TSNEModel", "TSNERequest", "TSNEResponse"]
