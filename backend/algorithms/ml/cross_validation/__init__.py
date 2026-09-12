"""
Cross-Validation algorithm module.

This module provides implementations of various cross-validation strategies
for model evaluation, including K-Fold, Stratified K-Fold, Shuffle Split,
Leave-One-Out, and Time Series Split.
"""

from .model import CrossValidationModel, train_cross_validation
from .schema import (
    CrossValidationRequest,
    CrossValidationResponse,
    CrossValidationMetrics,
    FoldMetrics,
    VisualizationData
)
from .data import load_dataset, get_dataset_info

__all__ = [
    'CrossValidationModel',
    'train_cross_validation',
    'CrossValidationRequest',
    'CrossValidationResponse',
    'CrossValidationMetrics',
    'FoldMetrics',
    'VisualizationData',
    'load_dataset',
    'get_dataset_info'
]
