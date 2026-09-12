"""Neural Style Transfer sample data management."""

import os
import urllib.request
from typing import Dict, Any, List
from pathlib import Path


# Sample content images (10 diverse images)
CONTENT_IMAGES = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/606px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "name": "content_landscape1.jpg",
        "description": "Mountain landscape"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg",
        "name": "content_landscape2.jpg",
        "description": "Alpine scene with lake"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/ReceivePhoto.jpeg/480px-ReceivePhoto.jpeg",
        "name": "content_portrait1.jpg",
        "description": "Person portrait"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg",
        "name": "content_portrait2.jpg",
        "description": "Cat portrait"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Duomo_di_Milano_dalla_Galleria.jpg/480px-Duomo_di_Milano_dalla_Galleria.jpg",
        "name": "content_architecture1.jpg",
        "description": "Milan cathedral"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Eiffel_tower_from_trocadero.jpg/427px-Eiffel_tower_from_trocadero.jpg",
        "name": "content_architecture2.jpg",
        "description": "Eiffel Tower"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/London_Bus_route_26_A.jpg/640px-London_Bus_route_26_A.jpg",
        "name": "content_urban.jpg",
        "description": "Urban street scene"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Domestic_goose_with_goslings_in_Prospect_Park_%2801976%29.jpg/640px-Domestic_goose_with_goslings_in_Prospect_Park_%2801976%29.jpg",
        "name": "content_nature.jpg",
        "description": "Nature scene with birds"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        "name": "content_outdoor.jpg",
        "description": "Outdoor path scene"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/La_Jolla_Aerial_Photo_by_D_Ramey_Logan.jpg/640px-La_Jolla_Aerial_Photo_by_D_Ramey_Logan.jpg",
        "name": "content_coastal.jpg",
        "description": "Coastal aerial view"
    }
]


# Sample style images (10 famous artistic styles)
STYLE_IMAGES = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/606px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "name": "style_starry_night.jpg",
        "description": "Van Gogh - Starry Night (post-impressionist swirls)"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Les_Demoiselles_d%27Avignon.jpg/480px-Les_Demoiselles_d%27Avignon.jpg",
        "name": "style_picasso.jpg",
        "description": "Picasso - Les Demoiselles d'Avignon (cubism)"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/440px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg",
        "name": "style_scream.jpg",
        "description": "Munch - The Scream (expressionism)"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Claude_Monet%2C_Impression%2C_soleil_levant.jpg/640px-Claude_Monet%2C_Impression%2C_soleil_levant.jpg",
        "name": "style_monet.jpg",
        "description": "Monet - Impression Sunrise (impressionism)"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/640px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
        "name": "style_kandinsky.jpg",
        "description": "Kandinsky - Composition VII (abstract)"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/The_Great_Wave_off_Kanagawa.jpg/640px-The_Great_Wave_off_Kanagawa.jpg",
        "name": "style_wave.jpg",
        "description": "Hokusai - The Great Wave (Japanese woodblock)"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Gustav_Klimt_016.jpg/449px-Gustav_Klimt_016.jpg",
        "name": "style_klimt.jpg",
        "description": "Klimt - The Kiss (art nouveau, gold)"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Piet_Mondrian%2C_1930_-_Mondrian_Composition_II_in_Red%2C_Blue%2C_and_Yellow.jpg/480px-Piet_Mondrian%2C_1930_-_Mondrian_Composition_II_in_Red%2C_Blue%2C_and_Yellow.jpg",
        "name": "style_mondrian.jpg",
        "description": "Mondrian - Composition (geometric)"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/El_coloso.jpg/396px-El_coloso.jpg",
        "name": "style_goya.jpg",
        "description": "Goya - The Colossus (romanticism)"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Vassily_Kandinsky%2C_1923_-_On_White_II.jpg/512px-Vassily_Kandinsky%2C_1923_-_On_White_II.jpg",
        "name": "style_abstract.jpg",
        "description": "Kandinsky - On White II (abstract expressionism)"
    }
]


def get_data_directory(subdir: str = "") -> Path:
    """Get or create the data directory for sample images.

    Args:
        subdir: Subdirectory name (e.g., 'content' or 'style')

    Returns:
        Path to the data directory
    """
    data_dir = Path(__file__).parent / "data"
    if subdir:
        data_dir = data_dir / subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_content_image(image_index: int = 0) -> str:
    """Download a content image.

    Args:
        image_index: Index of the content image to download

    Returns:
        Path to the downloaded image file

    Raises:
        ValueError: If image_index is out of range
        RuntimeError: If download fails
    """
    if image_index < 0 or image_index >= len(CONTENT_IMAGES):
        raise ValueError(f"Content image index must be between 0 and {len(CONTENT_IMAGES) - 1}")

    image_info = CONTENT_IMAGES[image_index]
    data_dir = get_data_directory("content")
    image_path = data_dir / image_info["name"]

    # Download if not already cached
    if not image_path.exists():
        try:
            print(f"Downloading content image: {image_info['name']}...")
            urllib.request.urlretrieve(image_info["url"], str(image_path))
            print(f"Downloaded to: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download content image: {str(e)}")

    return str(image_path)


def download_style_image(image_index: int = 0) -> str:
    """Download a style image.

    Args:
        image_index: Index of the style image to download

    Returns:
        Path to the downloaded image file

    Raises:
        ValueError: If image_index is out of range
        RuntimeError: If download fails
    """
    if image_index < 0 or image_index >= len(STYLE_IMAGES):
        raise ValueError(f"Style image index must be between 0 and {len(STYLE_IMAGES) - 1}")

    image_info = STYLE_IMAGES[image_index]
    data_dir = get_data_directory("style")
    image_path = data_dir / image_info["name"]

    # Download if not already cached
    if not image_path.exists():
        try:
            print(f"Downloading style image: {image_info['name']}...")
            urllib.request.urlretrieve(image_info["url"], str(image_path))
            print(f"Downloaded to: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download style image: {str(e)}")

    return str(image_path)


def get_available_content_images() -> List[Dict[str, Any]]:
    """Get list of available content images.

    Returns:
        List of dictionaries with image information
    """
    return [
        {
            "index": i,
            "name": img["name"],
            "description": img["description"]
        }
        for i, img in enumerate(CONTENT_IMAGES)
    ]


def get_available_style_images() -> List[Dict[str, Any]]:
    """Get list of available style images.

    Returns:
        List of dictionaries with image information
    """
    return [
        {
            "index": i,
            "name": img["name"],
            "description": img["description"]
        }
        for i, img in enumerate(STYLE_IMAGES)
    ]


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the style transfer dataset.

    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "Neural Style Transfer Sample Images",
        "description": "Curated content images and famous artistic styles for neural style transfer",
        "num_content_images": len(CONTENT_IMAGES),
        "num_style_images": len(STYLE_IMAGES),
        "content_categories": ["landscape", "portrait", "architecture", "urban", "nature"],
        "style_categories": ["post-impressionism", "cubism", "expressionism", "impressionism", "abstract", "art-nouveau"],
        "available_content_images": get_available_content_images(),
        "available_style_images": get_available_style_images(),
        "format": "RGB images",
        "source": "Wikimedia Commons (Public Domain)"
    }
