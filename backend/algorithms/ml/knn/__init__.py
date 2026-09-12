"""
K-Nearest Neighbors (KNN) algorithm package.

This package provides a complete implementation of the KNN classification
algorithm for the AI algorithms demonstration website.
"""

from .model import KNNModel, train_knn
from .schema import KNNRequest, KNNResponse, KNNMetrics, VisualizationData
from .data import load_and_prepare_data, get_iris_dataset

__all__ = [
    'KNNModel',
    'train_knn',
    'KNNRequest',
    'KNNResponse',
    'KNNMetrics',
    'VisualizationData',
    'load_and_prepare_data',
    'get_iris_dataset'
]
