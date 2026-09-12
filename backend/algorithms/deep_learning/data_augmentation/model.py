"""Data Augmentation model implementation using PIL and NumPy."""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import Dict, Any, List, Tuple
import time

from .schema import DataAugmentationRequest, DataAugmentationResponse, AugmentedImage
from .data import get_sample_image


class DataAugmentationModel:
    """Model for demonstrating data augmentation techniques.

    This class provides various image augmentation methods commonly used
    in deep learning to expand training datasets and improve model generalization.
    """

    def __init__(self, request: DataAugmentationRequest):
        """Initialize the data augmentation model.

        Args:
            request: DataAugmentationRequest with augmentation parameters
        """
        self.request = request
        self.random_state = request.random_state
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Get sample image
        self.original_image = get_sample_image()

    def augment(self) -> DataAugmentationResponse:
        """Apply data augmentation and generate results.

        Returns:
            DataAugmentationResponse with original and augmented images
        """
        start_time = time.time()

        # Generate augmented images
        augmented_images = []
        for i in range(self.request.num_augmented):
            # Randomly select augmentation types for this image
            num_transforms = random.randint(1, min(3, len(self.request.augmentation_types)))
            selected_transforms = random.sample(
                self.request.augmentation_types,
                num_transforms
            )

            # Apply augmentations
            augmented_img, descriptions = self._apply_random_augmentations(
                self.original_image.copy(),
                selected_transforms
            )

            augmented_images.append(AugmentedImage(
                index=i,
                image_data=augmented_img.tolist(),
                augmentations_applied=selected_transforms,
                description=", ".join(descriptions)
            ))

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Create visualization data
        visualization_data = self._create_visualization_data(augmented_images)

        # Create augmentation pipeline description
        augmentation_pipeline = {
            "total_augmentations": len(augmented_images),
            "techniques_available": [
                "rotation", "flip_horizontal", "flip_vertical",
                "brightness", "contrast", "blur", "crop", "scale",
                "noise", "color_jitter"
            ],
            "techniques_used": list(set(self.request.augmentation_types)),
            "random_combinations": True,
            "description": f"Generated {len(augmented_images)} augmented images using random combinations of {len(self.request.augmentation_types)} techniques"
        }

        return DataAugmentationResponse(
            success=True,
            original_image=self.original_image.tolist(),
            original_shape=list(self.original_image.shape),
            augmented_images=augmented_images,
            augmentation_pipeline=augmentation_pipeline,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            parameters_used={
                "augmentation_types": self.request.augmentation_types,
                "rotation_range": self.request.rotation_range,
                "brightness_factor": self.request.brightness_factor,
                "num_augmented": self.request.num_augmented,
                "random_state": self.request.random_state
            }
        )

    def _apply_random_augmentations(
        self,
        image: np.ndarray,
        transforms: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply a series of random augmentations to an image.

        Args:
            image: Input image as numpy array (H, W, C)
            transforms: List of transform names to apply

        Returns:
            Tuple of (augmented image, list of descriptions)
        """
        descriptions = []
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')

        for transform in transforms:
            pil_image, desc = apply_augmentation(
                pil_image,
                transform,
                rotation_range=self.request.rotation_range,
                brightness_factor=self.request.brightness_factor
            )
            descriptions.append(desc)

        return np.array(pil_image), descriptions

    def _create_visualization_data(self, augmented_images: List[AugmentedImage]) -> Dict[str, Any]:
        """Create visualization data for the frontend.

        Args:
            augmented_images: List of augmented images

        Returns:
            Dictionary with visualization data
        """
        # Determine grid layout
        num_images = len(augmented_images)
        if num_images <= 4:
            layout = "2x2"
        elif num_images <= 9:
            layout = "3x3"
        else:
            layout = "4x4"

        # Count augmentation types
        augmentation_counts = {}
        for aug_img in augmented_images:
            for aug_type in aug_img.augmentations_applied:
                augmentation_counts[aug_type] = augmentation_counts.get(aug_type, 0) + 1

        # Create statistics
        augmentation_stats = [
            {"type": aug_type, "count": count, "percentage": (count / num_images) * 100}
            for aug_type, count in augmentation_counts.items()
        ]
        augmentation_stats.sort(key=lambda x: x["count"], reverse=True)

        return {
            "gallery_layout": layout,
            "num_images": num_images,
            "augmentation_stats": augmentation_stats,
            "grid_size": {
                "rows": int(np.ceil(np.sqrt(num_images))),
                "cols": int(np.ceil(np.sqrt(num_images)))
            }
        }


def apply_augmentation(
    image: Image.Image,
    augmentation_type: str,
    rotation_range: float = 30.0,
    brightness_factor: float = 0.3
) -> Tuple[Image.Image, str]:
    """Apply a single augmentation to an image.

    Args:
        image: PIL Image to augment
        augmentation_type: Type of augmentation to apply
        rotation_range: Maximum rotation angle for rotation augmentation
        brightness_factor: Factor for brightness adjustment

    Returns:
        Tuple of (augmented image, description)
    """
    if augmentation_type == "rotation":
        angle = random.uniform(-rotation_range, rotation_range)
        augmented = image.rotate(angle, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
        return augmented, f"Rotated {angle:.1f}°"

    elif augmentation_type == "flip_horizontal":
        augmented = image.transpose(Image.FLIP_LEFT_RIGHT)
        return augmented, "Horizontal flip"

    elif augmentation_type == "flip_vertical":
        augmented = image.transpose(Image.FLIP_TOP_BOTTOM)
        return augmented, "Vertical flip"

    elif augmentation_type == "brightness":
        factor = 1.0 + random.uniform(-brightness_factor, brightness_factor)
        enhancer = ImageEnhance.Brightness(image)
        augmented = enhancer.enhance(factor)
        return augmented, f"Brightness ×{factor:.2f}"

    elif augmentation_type == "contrast":
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(image)
        augmented = enhancer.enhance(factor)
        return augmented, f"Contrast ×{factor:.2f}"

    elif augmentation_type == "blur":
        radius = random.uniform(0.5, 2.0)
        augmented = image.filter(ImageFilter.GaussianBlur(radius))
        return augmented, f"Blur (r={radius:.1f})"

    elif augmentation_type == "crop":
        width, height = image.size
        crop_size = random.uniform(0.7, 0.9)
        new_width = int(width * crop_size)
        new_height = int(height * crop_size)
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        augmented = image.crop((left, top, left + new_width, top + new_height))
        augmented = augmented.resize((width, height), Image.BICUBIC)
        return augmented, f"Random crop {int(crop_size*100)}%"

    elif augmentation_type == "scale":
        width, height = image.size
        scale_factor = random.uniform(0.8, 1.2)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        augmented = image.resize((new_width, new_height), Image.BICUBIC)
        # Crop or pad to original size
        if scale_factor > 1.0:
            # Crop center
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            augmented = augmented.crop((left, top, left + width, top + height))
        else:
            # Pad with gray
            new_img = Image.new('RGB', (width, height), (128, 128, 128))
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            new_img.paste(augmented, (left, top))
            augmented = new_img
        return augmented, f"Scale ×{scale_factor:.2f}"

    elif augmentation_type == "noise":
        img_array = np.array(image).astype(np.float32)
        noise_level = random.uniform(5, 20)
        noise = np.random.normal(0, noise_level, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        augmented = Image.fromarray(noisy_img, 'RGB')
        return augmented, f"Gaussian noise (σ={noise_level:.1f})"

    elif augmentation_type == "color_jitter":
        # Apply random color jitter
        img_array = np.array(image).astype(np.float32)
        # Random shifts for each channel
        r_shift = random.uniform(-20, 20)
        g_shift = random.uniform(-20, 20)
        b_shift = random.uniform(-20, 20)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + r_shift, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] + g_shift, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] + b_shift, 0, 255)
        augmented = Image.fromarray(img_array.astype(np.uint8), 'RGB')
        return augmented, f"Color jitter (R{r_shift:+.0f},G{g_shift:+.0f},B{b_shift:+.0f})"

    else:
        # Unknown augmentation type, return original
        return image, "No augmentation"
