"""Semantic Segmentation sample data management."""

import os
import urllib.request
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np


# PASCAL VOC 2012 classes (21 classes including background)
PASCAL_VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Color palette for PASCAL VOC classes (RGB)
PASCAL_VOC_COLORS = [
    [0, 0, 0],        # background - black
    [128, 0, 0],      # aeroplane - maroon
    [0, 128, 0],      # bicycle - green
    [128, 128, 0],    # bird - olive
    [0, 0, 128],      # boat - navy
    [128, 0, 128],    # bottle - purple
    [0, 128, 128],    # bus - teal
    [128, 128, 128],  # car - gray
    [64, 0, 0],       # cat - dark red
    [192, 0, 0],      # chair - red
    [64, 128, 0],     # cow - dark green
    [192, 128, 0],    # diningtable - yellow-green
    [64, 0, 128],     # dog - dark purple
    [192, 0, 128],    # horse - magenta
    [64, 128, 128],   # motorbike - dark cyan
    [192, 128, 128],  # person - pink
    [0, 64, 0],       # pottedplant - dark green
    [128, 64, 0],     # sheep - brown
    [0, 192, 0],      # sofa - lime
    [128, 192, 0],    # train - yellow
    [0, 64, 128]      # tvmonitor - dark blue
]


# Sample images for segmentation
SAMPLE_IMAGES = [
    {
        "url": "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg",
        "name": "dog1.jpg",
        "description": "Dog portrait for segmentation"
    },
    {
        "url": "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg",
        "name": "dog2.jpg",
        "description": "Dog on grass"
    },
    {
        "url": "https://ultralytics.com/images/bus.jpg",
        "name": "street.jpg",
        "description": "Street scene with vehicles"
    }
]


def get_data_directory() -> Path:
    """Get or create the data directory for segmentation sample images.

    Returns:
        Path to the data directory
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_sample_image(image_index: int = 0) -> str:
    """Download a sample image for segmentation.

    Args:
        image_index: Index of the sample image to download

    Returns:
        Path to the downloaded image file

    Raises:
        ValueError: If image_index is out of range
        RuntimeError: If download fails
    """
    if image_index < 0 or image_index >= len(SAMPLE_IMAGES):
        raise ValueError(f"Image index must be between 0 and {len(SAMPLE_IMAGES) - 1}")

    image_info = SAMPLE_IMAGES[image_index]
    data_dir = get_data_directory()
    image_path = data_dir / image_info["name"]

    # Download if not already cached
    if not image_path.exists():
        try:
            print(f"Downloading sample image: {image_info['name']}...")
            urllib.request.urlretrieve(image_info["url"], str(image_path))
            print(f"Downloaded to: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download image: {str(e)}")

    return str(image_path)


def get_available_images() -> List[Dict[str, Any]]:
    """Get list of available sample images.

    Returns:
        List of dictionaries with image information
    """
    return [
        {
            "index": i,
            "name": img["name"],
            "description": img["description"]
        }
        for i, img in enumerate(SAMPLE_IMAGES)
    ]


def get_class_info(num_classes: int = 21) -> Tuple[List[str], List[List[int]]]:
    """Get class names and colors for visualization.

    Args:
        num_classes: Number of classes to return

    Returns:
        Tuple of (class_names, class_colors)
    """
    # Use PASCAL VOC classes by default
    if num_classes <= len(PASCAL_VOC_CLASSES):
        return (
            PASCAL_VOC_CLASSES[:num_classes],
            PASCAL_VOC_COLORS[:num_classes]
        )

    # Generate additional classes if needed
    class_names = PASCAL_VOC_CLASSES.copy()
    class_colors = PASCAL_VOC_COLORS.copy()

    for i in range(len(PASCAL_VOC_CLASSES), num_classes):
        class_names.append(f"class_{i}")
        # Generate distinct colors using HSV color space
        hue = (i * 137) % 360  # Golden angle for better color distribution
        rgb = _hsv_to_rgb(hue / 360.0, 0.8, 0.9)
        class_colors.append(rgb)

    return class_names, class_colors


def _hsv_to_rgb(h: float, s: float, v: float) -> List[int]:
    """Convert HSV color to RGB.

    Args:
        h: Hue (0-1)
        s: Saturation (0-1)
        v: Value (0-1)

    Returns:
        RGB color as [r, g, b] integers (0-255)
    """
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [int(r * 255), int(g * 255), int(b * 255)]


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the segmentation dataset.

    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "PASCAL VOC 2012",
        "description": "Sample images for semantic segmentation demonstration",
        "num_classes": 21,
        "classes": PASCAL_VOC_CLASSES,
        "available_images": get_available_images(),
        "format": "DeepLab pretrained model",
        "source": "PyTorch Vision sample images"
    }
