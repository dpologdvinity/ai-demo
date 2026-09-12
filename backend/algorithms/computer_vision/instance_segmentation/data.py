"""Data management for Instance Segmentation."""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import urllib.request
import numpy as np


# COCO class names (80 classes + background)
COCO_CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Color palette for instance visualization (bright, distinct colors)
INSTANCE_COLORS = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
    [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
    [128, 0, 128], [0, 128, 128], [192, 0, 0], [0, 192, 0], [0, 0, 192],
    [192, 192, 0], [192, 0, 192], [0, 192, 192], [255, 128, 0], [255, 0, 128],
    [128, 255, 0], [0, 255, 128], [128, 0, 255], [0, 128, 255], [255, 128, 128],
    [128, 255, 128], [128, 128, 255], [255, 255, 128], [255, 128, 255], [128, 255, 255],
]

# Sample image URLs (diverse scenes with multiple instances)
# Using reliable sources that allow programmatic access
SAMPLE_IMAGE_URLS = [
    # 0: Street scene with bus and people
    "https://ultralytics.com/images/bus.jpg",
    # 1: Soccer players (multiple people)
    "https://ultralytics.com/images/zidane.jpg",
    # 2: City street with vehicles and pedestrians
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/street.jpg",
    # 3: Living room with furniture
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/living-room.jpg",
    # 4: Multiple people outdoors
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/group-people.jpg",
    # 5: Dogs in park
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/dogs.jpg",
    # 6: Horses in field
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/horses.jpg",
    # 7: Cats indoors
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/cats.jpg",
    # 8: Airport scene with airplanes
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/airport.jpg",
    # 9: Kitchen with appliances and objects
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/kitchen.jpg",
]

SAMPLE_IMAGE_DESCRIPTIONS = [
    "Street scene with bus and people",
    "Soccer players (multiple people)",
    "City street with vehicles and pedestrians",
    "Living room with furniture and objects",
    "Multiple people outdoors",
    "Dogs in park",
    "Horses in field",
    "Cats indoors",
    "Airport scene with airplanes",
    "Kitchen with appliances and objects",
]


def get_sample_images_dir() -> Path:
    """Get the directory where sample images are stored.

    Returns:
        Path to the sample images directory
    """
    current_dir = Path(__file__).parent
    sample_dir = current_dir / 'sample_images'
    sample_dir.mkdir(exist_ok=True)
    return sample_dir


def download_sample_image(image_index: int) -> str:
    """Download and cache a sample image.

    Args:
        image_index: Index of the sample image (0-9)

    Returns:
        Path to the downloaded image file

    Raises:
        ValueError: If image_index is out of range
        RuntimeError: If download fails
    """
    if image_index < 0 or image_index >= len(SAMPLE_IMAGE_URLS):
        raise ValueError(
            f"image_index must be between 0 and {len(SAMPLE_IMAGE_URLS) - 1}"
        )

    sample_dir = get_sample_images_dir()
    image_path = sample_dir / f"sample_{image_index}.jpg"

    # Download if not already cached
    if not image_path.exists():
        try:
            print(f"Downloading sample image {image_index}...")

            # Create request with headers to avoid 403 errors
            req = urllib.request.Request(
                SAMPLE_IMAGE_URLS[image_index],
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )

            with urllib.request.urlopen(req) as response:
                with open(image_path, 'wb') as f:
                    f.write(response.read())

            print(f"Downloaded to {image_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download sample image: {str(e)}")

    return str(image_path)


def get_instance_color(instance_id: int) -> List[int]:
    """Get a distinct color for an instance.

    Args:
        instance_id: Instance ID

    Returns:
        RGB color as [R, G, B]
    """
    return INSTANCE_COLORS[instance_id % len(INSTANCE_COLORS)]


def get_class_name(class_id: int) -> str:
    """Get class name from COCO class ID.

    Args:
        class_id: COCO class ID

    Returns:
        Class name string
    """
    if 0 <= class_id < len(COCO_CLASS_NAMES):
        return COCO_CLASS_NAMES[class_id]
    return f"class_{class_id}"


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the dataset.

    Returns:
        Dictionary with dataset metadata
    """
    return {
        "name": "COCO Instance Segmentation",
        "description": "Common Objects in Context (COCO) dataset with 80 object classes",
        "num_classes": 80,
        "num_samples": len(SAMPLE_IMAGE_URLS),
        "sample_descriptions": SAMPLE_IMAGE_DESCRIPTIONS,
        "class_names": [c for c in COCO_CLASS_NAMES if c != 'N/A'],
        "image_urls": SAMPLE_IMAGE_URLS
    }


def get_algorithm_info() -> Dict[str, Any]:
    """Get algorithm-specific information.

    Returns:
        Dictionary with algorithm details
    """
    return {
        "algorithm": "Mask R-CNN",
        "description": "Instance segmentation with pixel-level masks and bounding boxes",
        "framework": "PyTorch + torchvision",
        "pretrained_dataset": "COCO",
        "output_format": "Per-instance masks, bounding boxes, class labels, and confidence scores",
        "capabilities": [
            "Detect multiple instances of the same class",
            "Pixel-level segmentation masks",
            "Bounding box localization",
            "Confidence scoring per instance",
            "Handles overlapping objects"
        ]
    }
