"""Data loading and preparation for GAN.

This module provides functions to load the digits dataset and prepare it
for GAN training.
"""

from typing import Dict, Any
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_digits_data(
    random_state: int = 42
) -> Dict[str, Any]:
    """Load and prepare digits dataset for GAN training.

    Loads the 8x8 digits dataset from sklearn and normalizes pixel values
    to the range [0, 1] for GAN training.

    Args:
        random_state: Random seed for reproducibility. Default: 42

    Returns:
        Dictionary containing:
            - X: Feature array (numpy array, shape [n_samples, 64])
                 Images flattened from 8x8 to 64-dimensional vectors
            - images: Original 8x8 images (numpy array, shape [n_samples, 8, 8])
            - n_samples: Number of samples in the dataset
            - n_features: Number of features (64 for 8x8 images)
            - description: Dataset description string

    Raises:
        ValueError: If dataset loading fails.

    Example:
        >>> data = load_digits_data()
        >>> X = data['X']
        >>> print(f"Loaded {data['n_samples']} samples with {data['n_features']} features")
    """
    # Load digits dataset from sklearn
    digits = load_digits()

    # Get features and normalize to [0, 1] range (digits are in range 0-16)
    X = digits.data / 16.0

    # Get original images
    images = digits.images

    return {
        'X': X,
        'images': images,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'description': (
            f"Handwritten digits dataset with {X.shape[0]} samples. "
            f"Each sample is an 8x8 grayscale image (64 features) normalized to [0, 1]. "
            f"Used for training GAN to generate digit-like images."
        )
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the digits dataset.

    Provides metadata about the digits dataset used for GAN training
    without actually loading the data.

    Returns:
        Dictionary containing dataset metadata including type, dimensions,
        number of samples, and description.

    Example:
        >>> info = get_dataset_info()
        >>> print(f"Dataset: {info['name']}")
        >>> print(f"Samples: {info['n_samples']}, Features: {info['n_features']}")
    """
    return {
        'name': 'Digits Dataset (8x8)',
        'type': 'real-world',
        'n_samples': 1797,
        'n_features': 64,
        'image_shape': [8, 8],
        'pixel_range': [0.0, 1.0],
        'description': (
            'Handwritten digits dataset containing 8x8 grayscale images '
            'of digits 0-9. Each image is represented as 64 pixel values '
            'normalized to the range [0, 1]. This dataset is used to train '
            'a GAN to generate realistic digit-like images.'
        ),
        'use_case': (
            'Ideal for demonstrating Generative Adversarial Networks (GANs) '
            'and image generation tasks. The small image size (8x8) allows '
            'for fast training while still producing interpretable results.'
        )
    }


def prepare_data_for_gan(X: np.ndarray) -> np.ndarray:
    """Prepare data for GAN training.

    Ensures data is in the correct format and range for GAN training.
    GAN typically works best with data normalized to [0, 1] or [-1, 1].

    Args:
        X: Raw feature data, shape (n_samples, n_features)

    Returns:
        Prepared feature data in range [0, 1], ready for GAN training.

    Raises:
        ValueError: If X is empty or has wrong dimensionality.

    Example:
        >>> X = load_digits_data()['X']
        >>> X_prepared = prepare_data_for_gan(X)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")

    if X.size == 0:
        raise ValueError("X cannot be empty")

    # Ensure data is in [0, 1] range
    X_min, X_max = X.min(), X.max()
    if X_min < 0 or X_max > 1:
        # Normalize to [0, 1]
        X = (X - X_min) / (X_max - X_min + 1e-8)

    return X


def reshape_for_display(images: np.ndarray, grid_size: tuple = (4, 4)) -> np.ndarray:
    """Reshape generated images for display in a grid.

    Takes a batch of flattened images and arranges them in a grid for
    visualization purposes.

    Args:
        images: Array of flattened images, shape (n_images, 64)
        grid_size: Tuple (rows, cols) specifying grid dimensions.
            Default: (4, 4) for 16 images

    Returns:
        Grid of images arranged for display, shape (grid_h, grid_w)
        where grid_h = rows * 8 and grid_w = cols * 8

    Raises:
        ValueError: If number of images doesn't match grid_size.

    Example:
        >>> images = np.random.rand(16, 64)  # 16 images
        >>> grid = reshape_for_display(images, grid_size=(4, 4))
        >>> print(grid.shape)  # (32, 32) - 4x4 grid of 8x8 images
    """
    rows, cols = grid_size
    n_images = rows * cols

    if images.shape[0] < n_images:
        raise ValueError(
            f"Need at least {n_images} images for {rows}x{cols} grid, "
            f"got {images.shape[0]}"
        )

    # Take first n_images
    images = images[:n_images]

    # Reshape each image to 8x8
    images_2d = images.reshape(-1, 8, 8)

    # Arrange in grid
    grid = np.zeros((rows * 8, cols * 8))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            grid[i*8:(i+1)*8, j*8:(j+1)*8] = images_2d[idx]
            idx += 1

    return grid
