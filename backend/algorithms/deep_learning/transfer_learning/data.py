"""Data loading and preprocessing for Transfer Learning."""

import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def create_synthetic_image_dataset(
    dataset_name: str,
    num_classes: int = 5,
    samples_per_class: int = 30,
    img_size: int = 224,
    random_state: int = 42
) -> Dict[str, Any]:
    """Create a synthetic image dataset for transfer learning demonstration.

    Args:
        dataset_name: Name of dataset (flowers, animals, food)
        num_classes: Number of classes in dataset
        samples_per_class: Number of samples per class
        img_size: Image size (will create img_size x img_size x 3)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing dataset arrays and metadata
    """
    np.random.seed(random_state)

    # Total number of samples
    n_samples = num_classes * samples_per_class

    # Generate synthetic images (224x224x3 RGB images)
    # Each class will have slightly different characteristics
    images = []
    labels = []

    # Define class names based on dataset
    class_names_map = {
        'flowers': ['Rose', 'Daisy', 'Tulip', 'Sunflower', 'Orchid'],
        'animals': ['Cat', 'Dog', 'Bird', 'Fish', 'Rabbit'],
        'food': ['Pizza', 'Burger', 'Sushi', 'Salad', 'Pasta']
    }

    class_names = class_names_map.get(dataset_name, class_names_map['flowers'])[:num_classes]

    for class_idx in range(num_classes):
        for _ in range(samples_per_class):
            # Create synthetic image with class-specific patterns
            # Use different color distributions for different classes
            base_color = np.array([class_idx * 50, (num_classes - class_idx) * 40, 128])

            # Generate image with some randomness
            img = np.random.randn(img_size, img_size, 3) * 30 + base_color

            # Add some structure (horizontal/vertical patterns based on class)
            if class_idx % 2 == 0:
                # Add horizontal stripes
                for i in range(0, img_size, 20):
                    img[i:i+5, :, :] += 20
            else:
                # Add vertical stripes
                for i in range(0, img_size, 20):
                    img[:, i:i+5, :] += 20

            # Clip to valid range [0, 255]
            img = np.clip(img, 0, 255).astype(np.uint8)

            images.append(img)
            labels.append(class_idx)

    images = np.array(images)
    labels = np.array(labels)

    # Shuffle the data
    indices = np.random.permutation(n_samples)
    images = images[indices]
    labels = labels[indices]

    # Split into train/val/test (60/20/20)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)

    X_train = images[:n_train]
    y_train = labels[:n_train]
    X_val = images[n_train:n_train + n_val]
    y_val = labels[n_train:n_train + n_val]
    X_test = images[n_train + n_val:]
    y_test = labels[n_train + n_val:]

    logger.info(
        f"Created {dataset_name} dataset: "
        f"{len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples"
    )

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': class_names,
        'num_classes': num_classes,
        'img_size': img_size,
        'dataset_name': dataset_name,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test)
    }


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize images using ImageNet mean and std.

    Args:
        images: Array of images (N, H, W, C) with values in [0, 255]

    Returns:
        Normalized images
    """
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert to float and normalize to [0, 1]
    images = images.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    images = (images - mean) / std

    return images


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the transfer learning datasets.

    Returns:
        Dictionary containing dataset metadata
    """
    return {
        "name": "Small Custom Datasets",
        "description": "Synthetic small-scale image datasets for transfer learning demonstration",
        "available_datasets": [
            {
                "name": "flowers",
                "description": "5 flower types (Rose, Daisy, Tulip, Sunflower, Orchid)",
                "classes": 5,
                "samples": 150
            },
            {
                "name": "animals",
                "description": "5 animal types (Cat, Dog, Bird, Fish, Rabbit)",
                "classes": 5,
                "samples": 150
            },
            {
                "name": "food",
                "description": "5 food types (Pizza, Burger, Sushi, Salad, Pasta)",
                "classes": 5,
                "samples": 150
            }
        ],
        "split_ratio": "60% train, 20% validation, 20% test",
        "image_size": "224x224x3 (RGB)",
        "samples_per_class": 30,
        "total_samples": 150,
        "features": [
            "Small dataset size (typical transfer learning scenario)",
            "Balanced classes",
            "Synthetic data with class-specific patterns",
            "ImageNet-compatible preprocessing"
        ]
    }


def create_data_augmentation_transforms() -> Dict[str, Any]:
    """Define data augmentation transforms for training.

    Returns:
        Dictionary of augmentation configurations
    """
    return {
        "horizontal_flip": {"probability": 0.5},
        "rotation": {"degrees": 15},
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1
        },
        "random_crop": {"scale": (0.8, 1.0)},
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
