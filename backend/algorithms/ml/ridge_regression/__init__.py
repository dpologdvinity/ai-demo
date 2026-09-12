"""Ridge Regression algorithm implementation."""

from .model import RidgeRegressionModel
from .schema import (
    RidgeRegressionRequest,
    RidgeRegressionResponse,
)

__all__ = [
    "RidgeRegressionModel",
    "RidgeRegressionRequest",
    "RidgeRegressionResponse",
]
