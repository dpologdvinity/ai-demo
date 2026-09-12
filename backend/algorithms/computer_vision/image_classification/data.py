"""Image Classification sample data management."""

import os
import urllib.request
from typing import Dict, Any, List
from pathlib import Path


# Sample images for classification (URLs to publicly available images)
SAMPLE_IMAGES = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg",
        "name": "cat.jpg",
        "description": "Domestic cat portrait"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/640px-YellowLabradorLooking_new.jpg",
        "name": "dog.jpg",
        "description": "Golden Labrador"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Airbus_A380_blue_sky.jpg/640px-Airbus_A380_blue_sky.jpg",
        "name": "airplane.jpg",
        "description": "Commercial airplane in flight"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Ferrari_458_Italia_--_03-01-2012.jpg/640px-Ferrari_458_Italia_--_03-01-2012.jpg",
        "name": "car.jpg",
        "description": "Red sports car"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Domestic_goose_with_goslings_in_Prospect_Park_%2801976%29.jpg/640px-Domestic_goose_with_goslings_in_Prospect_Park_%2801976%29.jpg",
        "name": "bird.jpg",
        "description": "Goose with goslings"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Koala_climbing_tree.jpg/464px-Koala_climbing_tree.jpg",
        "name": "koala.jpg",
        "description": "Koala on eucalyptus tree"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Granny_smith_and_cross_section.jpg/480px-Granny_smith_and_cross_section.jpg",
        "name": "apple.jpg",
        "description": "Green apple with cross section"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/Honeycrisp.jpg/480px-Honeycrisp.jpg",
        "name": "apple2.jpg",
        "description": "Red apple close-up"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/London_Bus_route_26_A.jpg/640px-London_Bus_route_26_A.jpg",
        "name": "bus.jpg",
        "description": "Red double-decker bus"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Patern_test.jpg/640px-Patern_test.jpg",
        "name": "coffee.jpg",
        "description": "Cup of coffee with pattern"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Giant_panda_eating_bamboo.jpg/480px-Giant_panda_eating_bamboo.jpg",
        "name": "panda.jpg",
        "description": "Giant panda eating bamboo"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/African_Bush_Elephant.jpg/480px-African_Bush_Elephant.jpg",
        "name": "elephant.jpg",
        "description": "African elephant"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Red_strawberry.jpg/480px-Red_strawberry.jpg",
        "name": "strawberry.jpg",
        "description": "Fresh strawberry"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Pizza_Margherita_2.jpg/480px-Pizza_Margherita_2.jpg",
        "name": "pizza.jpg",
        "description": "Margherita pizza"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Bananavarieties.jpg/640px-Bananavarieties.jpg",
        "name": "banana.jpg",
        "description": "Bunch of bananas"
    }
]


def get_data_directory() -> Path:
    """Get or create the data directory for sample images.

    Returns:
        Path to the data directory
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_sample_image(image_index: int = 0) -> str:
    """Download a sample image for classification.

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
    """Get information about the ImageNet dataset.

    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "ImageNet Sample Images",
        "description": "Sample images for image classification demonstration using ImageNet-trained models",
        "num_classes": 1000,
        "available_images": get_available_images(),
        "format": "RGB images",
        "source": "Wikimedia Commons",
        "model_training": "Pre-trained on ImageNet-1K (1000 classes)"
    }
