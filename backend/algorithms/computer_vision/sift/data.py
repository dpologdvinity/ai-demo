"""SIFT sample data management."""

import os
import urllib.request
from typing import Dict, Any, List
from pathlib import Path


# Sample images with distinctive features for SIFT detection
# Buildings, objects with textures, and scenes with clear keypoints
SAMPLE_IMAGES = [
    {
        "url": "https://ultralytics.com/images/bus.jpg",
        "name": "bus.jpg",
        "description": "Street scene with vehicles - Excellent for feature detection"
    },
    {
        "url": "https://ultralytics.com/images/zidane.jpg",
        "name": "zidane.jpg",
        "description": "Sports scene - Good for matching and tracking"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesgray.jpg/640px-Bikesgray.jpg",
        "name": "bikes.jpg",
        "description": "Bicycles with mechanical structures - Good for matching"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Valve_original_%281%29.PNG/640px-Valve_original_%281%29.PNG",
        "name": "valve.png",
        "description": "Mechanical valve - High-contrast features"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/ReceiptSwiss.jpg/480px-ReceiptSwiss.jpg",
        "name": "receipt.jpg",
        "description": "Document with text features - OCR preprocessing"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Forestiera_Ossea_Rosace_Sud.jpg/640px-Forestiera_Ossea_Rosace_Sud.jpg",
        "name": "architecture.jpg",
        "description": "Gothic architecture with intricate patterns"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png/640px-Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png",
        "name": "cat.png",
        "description": "Natural texture with fur details"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Male_and_female_mallard_ducks.jpg/640px-Male_and_female_mallard_ducks.jpg",
        "name": "ducks.jpg",
        "description": "Natural scene with animals and water"
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box.png",
        "name": "box.png",
        "description": "Object with clear edges and corners"
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "name": "lena.jpg",
        "description": "Standard test image with varied features"
    }
]


def get_data_directory() -> Path:
    """Get or create the data directory for SIFT sample images.

    Returns:
        Path to the data directory
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_sample_image(image_index: int = 0) -> str:
    """Download a sample image for SIFT feature detection.

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


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the SIFT dataset.

    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "SIFT Feature Detection Sample Images",
        "description": "Collection of images with distinctive features for SIFT keypoint detection",
        "num_images": len(SAMPLE_IMAGES),
        "categories": [
            "Architecture & Buildings",
            "Mechanical Objects",
            "Natural Scenes",
            "Artwork & Paintings",
            "Documents",
            "Textured Objects"
        ],
        "available_images": get_available_images(),
        "use_cases": [
            "Image matching and registration",
            "Object recognition",
            "Panorama stitching",
            "3D reconstruction",
            "Augmented reality tracking",
            "Visual odometry"
        ],
        "source": "Public domain images from Wikimedia Commons"
    }
