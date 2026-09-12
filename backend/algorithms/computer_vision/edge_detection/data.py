"""Edge Detection sample data management."""

import os
import urllib.request
from typing import Dict, Any, List
from pathlib import Path


# Sample images for edge detection (various scenes for demonstration)
# Using reliable public image sources (Ultralytics and direct image hosts)
SAMPLE_IMAGES = [
    {
        "url": "https://ultralytics.com/images/bus.jpg",
        "name": "bus.jpg",
        "description": "Street scene - Vehicles, people, and urban elements"
    },
    {
        "url": "https://ultralytics.com/images/zidane.jpg",
        "name": "zidane.jpg",
        "description": "Sports scene - Soccer players with clear outlines"
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "name": "lena.jpg",
        "description": "Portrait - Classic image processing test image"
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
        "name": "building.jpg",
        "description": "Architecture - Building with geometric patterns"
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
        "name": "fruits.jpg",
        "description": "Still life - Fruits with varied textures"
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/chicky_512.png",
        "name": "chicky.png",
        "description": "Cartoon - High contrast illustration"
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
        "name": "messi.jpg",
        "description": "Sports portrait - Player in action"
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/home.jpg",
        "name": "home.jpg",
        "description": "Indoor scene - Interior with furniture and objects"
    }
]


def get_data_directory() -> Path:
    """Get or create the data directory for edge detection sample images.

    Returns:
        Path to the data directory
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_sample_image(image_index: int = 0) -> str:
    """Download a sample image for edge detection.

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
            # Add User-Agent header to avoid 403 errors
            req = urllib.request.Request(
                image_info["url"],
                headers={'User-Agent': 'Mozilla/5.0 (Edge Detection Demo)'}
            )
            with urllib.request.urlopen(req) as response, open(image_path, 'wb') as out_file:
                out_file.write(response.read())
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
    """Get information about the edge detection dataset.

    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "Edge Detection Sample Images",
        "description": "Diverse collection of images for demonstrating Canny edge detection",
        "num_images": len(SAMPLE_IMAGES),
        "categories": [
            "Natural scenes",
            "Mechanical objects",
            "Architecture",
            "Documents",
            "Portraits",
            "Sports/Action"
        ],
        "available_images": get_available_images(),
        "use_cases": [
            "Object detection preprocessing",
            "Image segmentation",
            "Feature extraction",
            "Medical imaging analysis",
            "Document scanning"
        ],
        "source": "Public domain images from Wikimedia Commons and Ultralytics"
    }
