"""Data utilities for ResNet: ImageNet labels and sample images."""

import json
from typing import Dict, List, Any
from PIL import Image
import numpy as np
import os
from pathlib import Path


def get_imagenet_labels() -> Dict[int, str]:
    """Get ImageNet class labels.

    Returns:
        Dictionary mapping class indices to human-readable labels
    """
    # ImageNet class labels (1000 classes)
    # This is a subset of common classes for demonstration
    # In production, load from a complete labels file
    labels = {
        0: "tench, Tinca tinca",
        1: "goldfish, Carassius auratus",
        2: "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
        # ... (truncated for brevity, in production include all 1000 classes)
        # For now, we'll return a comprehensive mapping
    }

    # Generate placeholder labels for all 1000 classes
    for i in range(1000):
        if i not in labels:
            labels[i] = f"Class {i}"

    return labels


def get_sample_images() -> List[Dict[str, Any]]:
    """Get sample images for testing.

    Returns:
        List of sample image data dictionaries
    """
    # Create synthetic sample images (solid colors for demonstration)
    # In production, these would be actual sample images from various categories
    sample_images = []

    # Define some sample image configurations
    samples = [
        {"name": "Cat", "color": (255, 200, 150), "description": "Orange tabby cat"},
        {"name": "Dog", "color": (200, 180, 160), "description": "Golden retriever"},
        {"name": "Bird", "color": (100, 150, 200), "description": "Blue bird"},
        {"name": "Car", "color": (180, 50, 50), "description": "Red sports car"},
        {"name": "Flower", "color": (255, 100, 150), "description": "Pink rose"},
        {"name": "Landscape", "color": (100, 180, 100), "description": "Green forest"},
        {"name": "Food", "color": (220, 180, 100), "description": "Pizza"},
        {"name": "Person", "color": (200, 150, 120), "description": "Person portrait"},
        {"name": "Building", "color": (150, 150, 150), "description": "Modern architecture"},
        {"name": "Animal", "color": (180, 140, 100), "description": "Wild animal"},
    ]

    for i, sample in enumerate(samples):
        # Create a synthetic gradient image
        img = create_sample_image(
            size=(224, 224),
            base_color=sample["color"],
            pattern="gradient"
        )

        sample_images.append({
            "index": i,
            "name": sample["name"],
            "description": sample["description"],
            "image": img
        })

    return sample_images


def create_sample_image(
    size: tuple = (224, 224),
    base_color: tuple = (128, 128, 128),
    pattern: str = "gradient"
) -> Image.Image:
    """Create a synthetic sample image.

    Args:
        size: Image size (width, height)
        base_color: Base RGB color
        pattern: Pattern type ('solid', 'gradient', 'noise')

    Returns:
        PIL Image
    """
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    if pattern == "solid":
        img_array[:, :] = base_color

    elif pattern == "gradient":
        # Create a gradient from base color to darker version
        for y in range(height):
            factor = y / height
            color = tuple(int(c * (1 - factor * 0.5)) for c in base_color)
            img_array[y, :] = color

    elif pattern == "noise":
        # Add noise to base color
        img_array[:, :] = base_color
        noise = np.random.randint(-30, 30, (height, width, 3))
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    else:
        img_array[:, :] = base_color

    return Image.fromarray(img_array, mode='RGB')


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for ResNet input.

    Args:
        image: PIL Image

    Returns:
        Preprocessed image array
    """
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to 224x224 (ResNet input size)
    image = image.resize((224, 224), Image.Resampling.LANCZOS)

    # Convert to array
    img_array = np.array(image).astype(np.float32)

    # Normalize (ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    img_array = (img_array - mean) / std

    return img_array


def postprocess_for_visualization(image: np.ndarray) -> np.ndarray:
    """Convert preprocessed image back to viewable format.

    Args:
        image: Preprocessed image array

    Returns:
        Image array in [0, 255] range
    """
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    img_array = (image * std) + mean

    # Clip to valid range
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    return img_array


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the dataset.

    Returns:
        Dataset information dictionary
    """
    return {
        "name": "ImageNet",
        "description": "Large-scale image classification dataset with 1000 classes",
        "num_classes": 1000,
        "num_samples": "1.2M training images, 50K validation images",
        "image_size": "224x224 (resized)",
        "categories": [
            "Animals (mammals, birds, fish, insects)",
            "Objects (furniture, vehicles, tools)",
            "Plants (flowers, trees)",
            "Scenes (landscapes, buildings)",
            "Food items"
        ],
        "preprocessing": [
            "Resize to 256x256",
            "Center crop to 224x224",
            "Normalize with ImageNet mean and std",
            "Mean: [0.485, 0.456, 0.406]",
            "Std: [0.229, 0.224, 0.225]"
        ]
    }


def load_imagenet_labels_from_file(filepath: str) -> Dict[int, str]:
    """Load ImageNet labels from a JSON file.

    Args:
        filepath: Path to labels JSON file

    Returns:
        Dictionary mapping class indices to labels

    Note:
        This is a helper for loading complete ImageNet labels from a file.
        The file should be in format: {"0": "label0", "1": "label1", ...}
    """
    try:
        with open(filepath, 'r') as f:
            labels_dict = json.load(f)
            # Convert string keys to integers
            return {int(k): v for k, v in labels_dict.items()}
    except FileNotFoundError:
        # Fall back to basic labels
        return get_imagenet_labels()


# Common ImageNet class labels (subset of most common/interesting classes)
COMMON_IMAGENET_CLASSES = {
    281: "tabby cat",
    282: "tiger cat",
    283: "Persian cat",
    284: "Siamese cat",
    285: "Egyptian cat",
    151: "Chihuahua",
    207: "golden retriever",
    235: "German shepherd",
    281: "tabby cat",
    340: "zebra",
    386: "African elephant",
    387: "Indian elephant",
    417: "balloon",
    483: "castle",
    510: "container ship",
    530: "dining table",
    609: "jersey",
    701: "parachute",
    717: "pizza",
    751: "purse",
    779: "school bus",
    817: "sports car",
    859: "tow truck",
    924: "guacamole",
    949: "strawberry",
    951: "lemon",
    985: "daisy"
}


def get_common_class_names() -> Dict[int, str]:
    """Get common/interesting ImageNet class names for display.

    Returns:
        Dictionary of common class indices to names
    """
    # Combine with full labels
    full_labels = get_imagenet_labels()
    full_labels.update(COMMON_IMAGENET_CLASSES)
    return full_labels
