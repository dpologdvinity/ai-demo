"""Pydantic schemas for Data Augmentation API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DataAugmentationRequest(BaseModel):
    """Request schema for data augmentation demonstration.

    Attributes:
        augmentation_types: List of augmentation types to apply
        rotation_range: Maximum rotation in degrees (for rotation augmentation)
        brightness_factor: Brightness adjustment factor (for brightness augmentation)
        num_augmented: Number of augmented copies to generate
        random_state: Random seed for reproducibility
    """

    augmentation_types: List[str] = Field(
        default=['rotation', 'flip_horizontal', 'brightness'],
        description="Augmentation types to apply (rotation, flip_horizontal, flip_vertical, brightness, contrast, blur, crop, scale, noise, color_jitter)"
    )
    rotation_range: float = Field(
        default=30.0,
        ge=0.0,
        le=180.0,
        description="Maximum rotation in degrees"
    )
    brightness_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Brightness adjustment factor (0.0 to 1.0)"
    )
    num_augmented: int = Field(
        default=9,
        ge=1,
        le=16,
        description="Number of augmented copies to generate"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "augmentation_types": ["rotation", "flip_horizontal", "brightness"],
                "rotation_range": 30.0,
                "brightness_factor": 0.3,
                "num_augmented": 9,
                "random_state": 42
            }
        }


class AugmentedImage(BaseModel):
    """Information about a single augmented image."""

    index: int = Field(description="Index of the augmented image")
    image_data: List[List[List[int]]] = Field(description="Image data as 3D array (H, W, C)")
    augmentations_applied: List[str] = Field(description="List of augmentations applied")
    description: str = Field(description="Human-readable description of augmentations")


class DataAugmentationResponse(BaseModel):
    """Response schema for data augmentation results.

    Attributes:
        success: Whether augmentation was successful
        original_image: Original image data (H, W, C)
        original_shape: Shape of original image [H, W, C]
        augmented_images: List of augmented images with metadata
        augmentation_pipeline: Description of the augmentation pipeline
        visualization_data: Data formatted for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        parameters_used: Parameters used for this augmentation run
    """

    success: bool = Field(
        description="Whether augmentation completed successfully"
    )
    original_image: List[List[List[int]]] = Field(
        description="Original image data (H, W, C)"
    )
    original_shape: List[int] = Field(
        description="Shape of original image [H, W, C]"
    )
    augmented_images: List[AugmentedImage] = Field(
        description="List of augmented images with metadata"
    )
    augmentation_pipeline: Dict[str, Any] = Field(
        description="Description of the augmentation pipeline used"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Data formatted for frontend visualization"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    parameters_used: Dict[str, Any] = Field(
        description="Parameters used for this augmentation run"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "original_shape": [224, 224, 3],
                "augmented_images": [
                    {
                        "index": 0,
                        "augmentations_applied": ["rotation"],
                        "description": "Rotated by 25.3 degrees"
                    }
                ],
                "augmentation_pipeline": {
                    "total_augmentations": 10,
                    "techniques_available": [
                        "rotation", "flip_horizontal", "flip_vertical",
                        "brightness", "contrast", "blur", "crop", "scale",
                        "noise", "color_jitter"
                    ]
                },
                "visualization_data": {
                    "gallery_layout": "3x3",
                    "augmentation_stats": {}
                },
                "execution_time_ms": 156.78,
                "parameters_used": {
                    "augmentation_types": ["rotation", "flip_horizontal", "brightness"],
                    "rotation_range": 30.0,
                    "brightness_factor": 0.3,
                    "num_augmented": 9
                }
            }
        }
