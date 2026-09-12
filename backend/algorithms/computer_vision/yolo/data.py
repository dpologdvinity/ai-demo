"""YOLO Object Detection sample data management."""

import os
import urllib.request
from typing import Dict, Any, List
from pathlib import Path


# Sample images for object detection (URLs to publicly available images)
SAMPLE_IMAGES = [
    {
        "url": "https://ultralytics.com/images/bus.jpg",
        "name": "bus.jpg",
        "description": "Street scene with bus and people"
    },
    {
        "url": "https://ultralytics.com/images/zidane.jpg",
        "name": "zidane.jpg",
        "description": "Soccer players on field"
    },
    {
        "url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
        "name": "zidane_alternate.jpg",
        "description": "Sports action with multiple people"
    },
    {
        "url": "https://images.pexels.com/photos/3771089/pexels-photo-3771089.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "traffic.jpg",
        "description": "Traffic scene with cars and motorcycles"
    },
    {
        "url": "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "city_street.jpg",
        "description": "Urban street with vehicles and pedestrians"
    },
    {
        "url": "https://images.pexels.com/photos/169647/pexels-photo-169647.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "cars_parking.jpg",
        "description": "Parking lot with multiple cars"
    },
    {
        "url": "https://images.pexels.com/photos/1181406/pexels-photo-1181406.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "dogs_park.jpg",
        "description": "Dogs playing in park"
    },
    {
        "url": "https://images.pexels.com/photos/704569/pexels-photo-704569.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "living_room.jpg",
        "description": "Indoor living room scene with furniture"
    },
    {
        "url": "https://images.pexels.com/photos/1805164/pexels-photo-1805164.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "kitchen.jpg",
        "description": "Kitchen interior with appliances"
    },
    {
        "url": "https://images.pexels.com/photos/2662116/pexels-photo-2662116.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "horses.jpg",
        "description": "Horses in field"
    },
    {
        "url": "https://images.pexels.com/photos/15286/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=640",
        "name": "laptop_desk.jpg",
        "description": "Office desk with laptop and accessories"
    },
    {
        "url": "https://images.pexels.com/photos/1661535/pexels-photo-1661535.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "airport.jpg",
        "description": "Airport scene with airplanes"
    },
    {
        "url": "https://images.pexels.com/photos/1059078/pexels-photo-1059078.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "beach.jpg",
        "description": "Beach scene with people and umbrellas"
    },
    {
        "url": "https://images.pexels.com/photos/2101137/pexels-photo-2101137.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "cats.jpg",
        "description": "Multiple cats indoors"
    },
    {
        "url": "https://images.pexels.com/photos/1660995/pexels-photo-1660995.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "train_station.jpg",
        "description": "Train station with train and people"
    },
    {
        "url": "https://images.pexels.com/photos/1181534/pexels-photo-1181534.jpeg?auto=compress&cs=tinysrgb&w=640",
        "name": "birds.jpg",
        "description": "Birds on branches in outdoor setting"
    }
]


def get_data_directory() -> Path:
    """Get or create the data directory for YOLO sample images.

    Returns:
        Path to the data directory
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_sample_image(image_index: int = 0) -> str:
    """Download a sample image for object detection.

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
    """Get information about the YOLO dataset.

    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "COCO Sample Images",
        "description": "Sample images for YOLO object detection demonstration",
        "num_classes": 80,
        "classes": [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ],
        "available_images": get_available_images(),
        "format": "COCO format",
        "source": "Ultralytics sample images"
    }
