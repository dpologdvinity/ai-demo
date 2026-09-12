"""Face Detection sample data management."""

import os
import urllib.request
from typing import Dict, Any, List
from pathlib import Path


# Sample images with faces for face detection
SAMPLE_IMAGES = [
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "name": "lena.jpg",
        "description": "Classic portrait - Single face"
    },
    {
        "url": "https://images.pexels.com/photos/1595385/pexels-photo-1595385.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "group1.jpg",
        "description": "Small group - 2-3 faces"
    },
    {
        "url": "https://images.pexels.com/photos/1157394/pexels-photo-1157394.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "group2.jpg",
        "description": "Family group - 3-5 faces"
    },
    {
        "url": "https://images.pexels.com/photos/1464820/pexels-photo-1464820.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "business.jpg",
        "description": "Business setting - Multiple faces"
    },
    {
        "url": "https://images.pexels.com/photos/1206059/pexels-photo-1206059.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "outdoor.jpg",
        "description": "Outdoor portrait - Natural lighting"
    },
    {
        "url": "https://images.pexels.com/photos/1181690/pexels-photo-1181690.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "closeup.jpg",
        "description": "Close-up portrait - Large face"
    },
    {
        "url": "https://images.pexels.com/photos/1181424/pexels-photo-1181424.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "studio.jpg",
        "description": "Studio portrait - Professional lighting"
    },
    {
        "url": "https://images.pexels.com/photos/1496648/pexels-photo-1496648.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "diverse.jpg",
        "description": "Diverse group - Multiple ethnicities"
    }
]


def get_data_directory() -> Path:
    """Get or create the data directory for face detection sample images.

    Returns:
        Path to the data directory
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_sample_image(image_index: int = 0) -> str:
    """Download a sample image for face detection.

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
            # Add user agent to avoid 403 errors from Pexels
            req = urllib.request.Request(
                image_info["url"],
                headers={'User-Agent': 'Mozilla/5.0'}
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
    """Get information about the face detection dataset.

    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "Face Detection Sample Images",
        "description": "Diverse collection of portrait and group photos for face detection",
        "num_images": len(SAMPLE_IMAGES),
        "categories": [
            "Single portraits",
            "Small groups (2-5 people)",
            "Large groups (5+ people)",
            "Various lighting conditions",
            "Different ethnicities and ages"
        ],
        "available_images": get_available_images(),
        "use_cases": [
            "Photo organization and tagging",
            "Security and surveillance systems",
            "Attendance tracking",
            "Social media auto-tagging",
            "Demographics analysis"
        ],
        "source": "Public domain and free stock images"
    }
