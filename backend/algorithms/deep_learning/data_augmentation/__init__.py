"""Data Augmentation demonstration module.

This module provides various image augmentation techniques commonly used in
deep learning to expand training datasets and improve model generalization.
"""

from .schema import (
    DataAugmentationRequest,
    DataAugmentationResponse,
    AugmentedImage
)
from .model import DataAugmentationModel, apply_augmentation
from .data import get_dataset_info, get_sample_image

__all__ = [
    "DataAugmentationRequest",
    "DataAugmentationResponse",
    "AugmentedImage",
    "DataAugmentationModel",
    "apply_augmentation",
    "get_dataset_info",
    "get_sample_image"
]
