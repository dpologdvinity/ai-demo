"""Data utilities for Data Augmentation: sample images and dataset info."""

import numpy as np
from typing import Dict, Any
from PIL import Image


def get_sample_image(size: tuple = (224, 224)) -> np.ndarray:
    """Get a sample image for augmentation demonstration.

    Creates a synthetic sample image with various visual features (colors,
    gradients, shapes) to make augmentation effects clearly visible.

    Args:
        size: Image size (width, height)

    Returns:
        Image array in uint8 format with shape (H, W, 3)
    """
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a colorful gradient background
    for i in range(height):
        for j in range(width):
            # Create a radial gradient from center
            center_x, center_y = width // 2, height // 2
            dist_x = (j - center_x) / width
            dist_y = (i - center_y) / height
            dist = np.sqrt(dist_x**2 + dist_y**2)

            # Create color channels based on position
            r = int(128 + 127 * np.sin(dist * 10))
            g = int(128 + 127 * np.cos(dist * 8))
            b = int(128 + 127 * np.sin(dist * 12 + np.pi / 4))

            img_array[i, j] = [r, g, b]

    # Add geometric shapes to make transformations visible
    # Circle in the center
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 6
    y_coords, x_coords = np.ogrid[:height, :width]
    mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
    img_array[mask] = [255, 200, 50]  # Yellow circle

    # Add a rectangle in the top-left
    rect_height, rect_width = height // 5, width // 5
    img_array[10:10+rect_height, 10:10+rect_width] = [50, 150, 255]  # Blue rectangle

    # Add a triangle in the bottom-right (using simple filled triangle)
    triangle_size = min(width, height) // 6
    tri_x, tri_y = width - triangle_size - 20, height - triangle_size - 20
    for i in range(triangle_size):
        for j in range(triangle_size - i):
            if tri_y + i < height and tri_x + j < width:
                img_array[tri_y + i, tri_x + j] = [255, 100, 150]  # Pink triangle

    # Add text-like pattern (horizontal lines) on the left side
    for i in range(height // 4, height // 2, 15):
        if i + 3 < height:
            img_array[i:i+3, width//10:width//3] = [255, 255, 255]  # White lines

    return img_array


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the dataset used for data augmentation.

    Returns:
        Dictionary containing dataset metadata
    """
    return {
        "name": "Sample Image",
        "description": "Synthetic sample image with various features to demonstrate augmentation effects",
        "size": "224x224 pixels",
        "channels": 3,
        "format": "RGB",
        "features": [
            "Colorful gradient background",
            "Geometric shapes (circle, rectangle, triangle)",
            "Text-like patterns",
            "Various colors for clear augmentation visualization"
        ],
        "purpose": "Demonstrate how data augmentation transforms images",
        "notes": [
            "Sample image is synthetically generated",
            "Designed to make augmentation effects clearly visible",
            "Real applications would use actual training images"
        ]
    }
