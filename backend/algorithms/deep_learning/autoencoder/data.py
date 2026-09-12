"""Data loading and preparation for Autoencoder.

This module provides functions to load MNIST digits dataset and prepare it
for autoencoder training.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE


def load_mnist_data(
    n_samples: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """Load and prepare MNIST digits dataset for autoencoder training.

    Loads the 8x8 digits dataset from sklearn (subset of MNIST) and splits
    into train/test sets. Images are normalized to [0, 1] range.

    Args:
        n_samples: Number of samples to use (for speed). Default: 1000
        test_size: Fraction of data to use for testing. Default: 0.2
        random_state: Random seed for reproducibility. Default: 42

    Returns:
        Dictionary containing:
            - X_train: Training features (n_train, 64)
            - X_test: Test features (n_test, 64)
            - y_train: Training labels (n_train,)
            - y_test: Test labels (n_test,)
            - n_samples: Total number of samples
            - n_features: Number of features (64)
            - description: Dataset description

    Example:
        >>> data = load_mnist_data(n_samples=500)
        >>> X_train = data['X_train']
        >>> print(f"Training samples: {X_train.shape[0]}")
    """
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Normalize to [0, 1] range (digits are in range 0-16)
    X = X / 16.0

    # Use subset for faster training
    if n_samples < len(X):
        indices = np.random.RandomState(random_state).choice(
            len(X), n_samples, replace=False
        )
        X = X[indices]
        y = y[indices]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'description': (
            f"MNIST digits dataset (8x8) with {len(X)} samples. "
            f"Each sample is an 8x8 grayscale image (64 features) normalized to [0, 1]. "
            f"Used for unsupervised autoencoder training."
        )
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the MNIST digits dataset.

    Provides metadata about the dataset used for autoencoder training
    without actually loading the data.

    Returns:
        Dictionary containing dataset metadata including type, dimensions,
        number of samples, and description.

    Example:
        >>> info = get_dataset_info()
        >>> print(f"Dataset: {info['name']}")
    """
    return {
        'name': 'MNIST Digits (8x8)',
        'type': 'real-world',
        'n_samples': 1797,
        'n_features': 64,
        'image_shape': [8, 8],
        'pixel_range': [0.0, 1.0],
        'n_classes': 10,
        'description': (
            'Handwritten digits dataset containing 8x8 grayscale images '
            'of digits 0-9. Each image is represented as 64 pixel values '
            'normalized to the range [0, 1]. This dataset is used to train '
            'an autoencoder for dimensionality reduction and reconstruction.'
        ),
        'use_case': (
            'Ideal for demonstrating autoencoders and unsupervised learning. '
            'The autoencoder learns to compress images into a low-dimensional '
            'latent space and reconstruct them, discovering important features '
            'without supervision.'
        )
    }


def compute_latent_visualization(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    latent_dim: int,
    random_state: int = 42
) -> Tuple[np.ndarray, str]:
    """Compute 2D coordinates for latent space visualization.

    If latent_dim is 2, returns the latent vectors directly.
    Otherwise, uses t-SNE to project to 2D for visualization.

    Args:
        latent_vectors: Latent space representations (n_samples, latent_dim)
        labels: Class labels for coloring (n_samples,)
        latent_dim: Dimensionality of latent space
        random_state: Random seed for t-SNE. Default: 42

    Returns:
        Tuple of (coordinates_2d, projection_method)
            - coordinates_2d: 2D coordinates (n_samples, 2)
            - projection_method: 'none' if latent_dim=2, 'tsne' otherwise

    Example:
        >>> latent = np.random.randn(100, 32)
        >>> labels = np.random.randint(0, 10, 100)
        >>> coords, method = compute_latent_visualization(latent, labels, 32)
        >>> print(f"Method: {method}, Shape: {coords.shape}")
    """
    if latent_dim == 2:
        # Already 2D, no projection needed
        return latent_vectors, 'none'
    else:
        # Use t-SNE to project to 2D
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=min(30, len(latent_vectors) - 1)
        )
        coords_2d = tsne.fit_transform(latent_vectors)
        return coords_2d, 'tsne'


def prepare_sample_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sample images for original vs reconstructed comparison.

    Args:
        original: Original images (n_total, 64)
        reconstructed: Reconstructed images (n_total, 64)
        n_samples: Number of samples to return. Default: 10

    Returns:
        Tuple of (original_samples, reconstructed_samples)
            Both are arrays of shape (n_samples, 64)

    Example:
        >>> orig = np.random.rand(100, 64)
        >>> recon = np.random.rand(100, 64)
        >>> orig_sample, recon_sample = prepare_sample_comparison(orig, recon, 5)
        >>> print(f"Samples: {orig_sample.shape}")
    """
    n_available = min(len(original), len(reconstructed))
    n_samples = min(n_samples, n_available)

    return original[:n_samples], reconstructed[:n_samples]
