"""Data utilities for Convolutional Layers demonstration."""

import numpy as np
from typing import Dict, Any


def load_sample_image() -> np.ndarray:
    """Load a sample image for convolution demonstration.

    Creates a synthetic image with various features (edges, patterns, shapes)
    to demonstrate how different filters respond.

    Returns:
        2D numpy array representing grayscale image (values 0-1)
    """
    # Create 28x28 image with interesting patterns
    size = 28
    image = np.zeros((size, size), dtype=np.float32)

    # Add a square (for edge detection)
    image[8:20, 8:20] = 1.0

    # Add a diagonal line
    for i in range(10):
        if 18 + i < size and 5 + i < size:
            image[18 + i, 5 + i] = 0.8

    # Add some texture (gradient)
    for i in range(5, 15):
        for j in range(20, 26):
            if i < size and j < size:
                image[i, j] = (i - 5) / 10.0

    # Add a circle-like pattern
    center_x, center_y = 22, 16
    radius = 4
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if dist < radius:
                image[i, j] = max(image[i, j], 0.6)

    # Add some noise for texture
    noise = np.random.RandomState(42).uniform(-0.1, 0.1, (size, size))
    image = np.clip(image + noise, 0, 1).astype(np.float32)

    return image


def get_common_filters() -> Dict[str, np.ndarray]:
    """Get common predefined convolution filters.

    Returns:
        Dictionary mapping filter names to kernel arrays
    """
    filters = {}

    # Sobel X (vertical edge detection)
    filters['sobel_x'] = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Sobel Y (horizontal edge detection)
    filters['sobel_y'] = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    # Gaussian Blur (3x3)
    filters['gaussian_blur'] = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0

    # Sharpen
    filters['sharpen'] = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # Edge detection (Laplacian)
    filters['edge_detect'] = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    return filters


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the dataset used.

    Returns:
        Dictionary with dataset metadata
    """
    return {
        'name': 'Sample Image',
        'description': 'Synthetic grayscale image with various features for demonstrating convolution',
        'image_size': '28x28 pixels',
        'channels': 1,
        'features': [
            'Rectangular shapes for edge detection',
            'Diagonal lines for orientation-specific filters',
            'Gradients for texture analysis',
            'Circular patterns for radial features',
            'Noise for realistic texture'
        ],
        'value_range': '0.0 to 1.0 (normalized)'
    }


def get_filter_descriptions() -> Dict[str, Dict[str, str]]:
    """Get descriptions of common filters and their uses.

    Returns:
        Dictionary mapping filter names to descriptions
    """
    return {
        'sobel_x': {
            'name': 'Sobel X',
            'description': 'Detects vertical edges by computing horizontal gradient',
            'use_case': 'Finding vertical boundaries in images'
        },
        'sobel_y': {
            'name': 'Sobel Y',
            'description': 'Detects horizontal edges by computing vertical gradient',
            'use_case': 'Finding horizontal boundaries in images'
        },
        'gaussian_blur': {
            'name': 'Gaussian Blur',
            'description': 'Smooths image by averaging nearby pixels with Gaussian weights',
            'use_case': 'Noise reduction and smoothing before further processing'
        },
        'sharpen': {
            'name': 'Sharpen',
            'description': 'Enhances edges and fine details by emphasizing high-frequency components',
            'use_case': 'Improving image clarity and emphasizing details'
        },
        'edge_detect': {
            'name': 'Edge Detection (Laplacian)',
            'description': 'Detects edges in all directions using second derivative',
            'use_case': 'Finding all edges regardless of orientation'
        }
    }
