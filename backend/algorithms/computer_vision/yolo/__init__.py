"""YOLO Object Detection implementation."""

from .model import YOLOModel
from .schema import YOLORequest, YOLOResponse

__all__ = ['YOLOModel', 'YOLORequest', 'YOLOResponse']
