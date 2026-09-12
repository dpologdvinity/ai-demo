"""Imbalanced Classification module.

This module provides tools for handling class imbalance in classification
tasks using techniques like SMOTE, undersampling, oversampling, and class weights.
"""

from .model import ImbalancedClassificationModel
from .schema import (
    ImbalancedClassificationRequest,
    ImbalancedClassificationResponse,
    ImbalancedClassificationMetrics,
    StrategyMetrics,
    VisualizationData
)
from .data import prepare_imbalanced_data, get_dataset_info

__all__ = [
    'ImbalancedClassificationModel',
    'ImbalancedClassificationRequest',
    'ImbalancedClassificationResponse',
    'ImbalancedClassificationMetrics',
    'StrategyMetrics',
    'VisualizationData',
    'prepare_imbalanced_data',
    'get_dataset_info'
]
