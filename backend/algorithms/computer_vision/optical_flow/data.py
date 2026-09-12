"""Optical Flow sample data management."""

import os
import urllib.request
from typing import Dict, Any, List, Tuple
from pathlib import Path


# Sample image pairs for optical flow (consecutive frames showing movement)
SAMPLE_IMAGE_PAIRS = [
    {
        "frame1_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesgray.jpg/640px-Bikesgray.jpg",
        "frame2_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesgray.jpg/640px-Bikesgray.jpg",
        "name": "bikes_static",
        "description": "Static scene - Minimal motion (baseline test)",
        "motion_type": "static"
    },
    {
        "frame1_url": "https://ultralytics.com/images/bus.jpg",
        "frame2_url": "https://ultralytics.com/images/zidane.jpg",
        "name": "urban_scene",
        "description": "Urban scene transition - Camera/scene motion",
        "motion_type": "camera_motion"
    },
    {
        "frame1_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Male_and_female_mallard_ducks.jpg/640px-Male_and_female_mallard_ducks.jpg",
        "frame2_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png/640px-Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png",
        "name": "nature_transition",
        "description": "Natural scene transition - Object appearance change",
        "motion_type": "scene_change"
    },
]

# For demo purposes with real motion, we'll also synthetically generate motion
# by applying transformations to single images
SYNTHETIC_MOTION_IMAGES = [
    {
        "url": "https://ultralytics.com/images/bus.jpg",
        "name": "bus_synthetic.jpg",
        "description": "Bus scene - Simulated rightward motion",
        "motion_type": "horizontal_translation"
    },
    {
        "url": "https://ultralytics.com/images/zidane.jpg",
        "name": "sports_synthetic.jpg",
        "description": "Sports scene - Simulated camera pan",
        "motion_type": "mixed_translation"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Male_and_female_mallard_ducks.jpg/640px-Male_and_female_mallard_ducks.jpg",
        "name": "ducks_synthetic.jpg",
        "description": "Duck scene - Simulated zoom motion",
        "motion_type": "zoom"
    },
]


def get_data_directory() -> Path:
    """Get or create the data directory for optical flow sample images.

    Returns:
        Path to the data directory
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_sample_image_pair(image_pair_index: int = 0) -> Tuple[str, str]:
    """Download a sample image pair or single image for synthetic motion.

    Args:
        image_pair_index: Index of the sample image pair to download

    Returns:
        Tuple of (frame1_path, frame2_path or motion_type)

    Raises:
        ValueError: If image_pair_index is out of range
        RuntimeError: If download fails
    """
    total_samples = len(SYNTHETIC_MOTION_IMAGES)

    if image_pair_index < 0 or image_pair_index >= total_samples:
        raise ValueError(f"Image pair index must be between 0 and {total_samples - 1}")

    # Use synthetic motion approach for better demo
    image_info = SYNTHETIC_MOTION_IMAGES[image_pair_index]
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

    # Return path and motion type (will be used to generate synthetic frame2)
    return str(image_path), image_info["motion_type"]


def get_available_image_pairs() -> List[Dict[str, Any]]:
    """Get list of available sample image pairs.

    Returns:
        List of dictionaries with image pair information
    """
    return [
        {
            "index": i,
            "name": img["name"],
            "description": img["description"],
            "motion_type": img["motion_type"]
        }
        for i, img in enumerate(SYNTHETIC_MOTION_IMAGES)
    ]


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the optical flow dataset.

    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "Optical Flow Sample Frames",
        "description": "Image pairs with synthetic motion for demonstrating optical flow algorithms",
        "num_pairs": len(SYNTHETIC_MOTION_IMAGES),
        "motion_types": [
            "Horizontal translation",
            "Mixed translation",
            "Zoom motion",
        ],
        "available_pairs": get_available_image_pairs(),
        "algorithms": ["Farneback Dense Flow", "Lucas-Kanade Sparse Flow"],
        "use_cases": [
            "Video stabilization",
            "Object tracking",
            "Motion detection",
            "Autonomous navigation",
            "Sports analysis"
        ],
        "source": "Public domain images with synthetic motion transformations"
    }
