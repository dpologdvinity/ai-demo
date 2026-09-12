"""Instance Segmentation (Mask R-CNN) implementation."""

from .model import InstanceSegmentationModel
from .schema import InstanceSegmentationRequest, InstanceSegmentationResponse

__all__ = ['InstanceSegmentationModel', 'InstanceSegmentationRequest', 'InstanceSegmentationResponse']
