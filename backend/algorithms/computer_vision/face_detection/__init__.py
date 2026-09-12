"""Face Detection using Haar Cascades module."""

from .model import FaceDetectionModel
from .schema import FaceDetectionRequest, FaceDetectionResponse, FaceStatistics

__all__ = [
    'FaceDetectionModel',
    'FaceDetectionRequest',
    'FaceDetectionResponse',
    'FaceStatistics',
]
