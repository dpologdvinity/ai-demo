"""Data loading and preprocessing utilities for Autoencoder Variants."""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_mnist_data(
    n_samples: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True
) -> Dict[str, Any]:
    """Load MNIST digits dataset (8x8 digits from sklearn).

    Args:
        n_samples: Number of samples to load (max 1797). Default: 1000
        test_size: Proportion of test set. Default: 0.2
        random_state: Random seed. Default: 42
        normalize: Whether to normalize to [0, 1]. Default: True

    Returns:
        Dictionary containing:
            - X_train: Training data, shape (n_train, 64)
            - X_test: Test data, shape (n_test, 64)
            - y_train: Training labels
            - y_test: Test labels
            - n_features: Number of features (64)
            - image_shape: Shape of images (8, 8)
    """
    # Load digits dataset (8x8 grayscale images, 1797 samples, 10 classes)
    digits = load_digits()
    X = digits.data[:n_samples]
    y = digits.target[:n_samples]

    # Normalize to [0, 1] if requested
    if normalize:
        X = X / 16.0  # digits are in [0, 16] range

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        'X_train': X_train.astype(np.float32),
        'X_test': X_test.astype(np.float32),
        'y_train': y_train,
        'y_test': y_test,
        'n_features': X_train.shape[1],
        'image_shape': (8, 8)
    }


def add_noise(X: np.ndarray, noise_factor: float = 0.3) -> np.ndarray:
    """Add Gaussian noise to images for denoising autoencoder.

    Args:
        X: Clean images, shape (n_samples, 64)
        noise_factor: Standard deviation of Gaussian noise. Default: 0.3

    Returns:
        Noisy images, clipped to [0, 1], shape (n_samples, 64)
    """
    noise = np.random.normal(0, noise_factor, X.shape)
    X_noisy = X + noise
    X_noisy = np.clip(X_noisy, 0, 1)  # Keep in valid range
    return X_noisy.astype(np.float32)


def compute_latent_visualization(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    latent_dim: int,
    random_state: int = 42
) -> Tuple[np.ndarray, str]:
    """Compute 2D coordinates for latent space visualization.

    If latent_dim == 2, use the latent space directly.
    Otherwise, use PCA or t-SNE to project to 2D.

    Args:
        latent_vectors: Latent representations, shape (n_samples, latent_dim)
        labels: Class labels for coloring
        latent_dim: Dimension of latent space
        random_state: Random seed. Default: 42

    Returns:
        Tuple of (coords_2d, projection_method):
            - coords_2d: 2D coordinates, shape (n_samples, 2)
            - projection_method: 'none', 'pca', or 'tsne'
    """
    if latent_dim == 2:
        # Already 2D, use directly
        return latent_vectors, 'none'
    elif latent_dim <= 50:
        # Use PCA for moderate dimensions
        pca = PCA(n_components=2, random_state=random_state)
        coords_2d = pca.fit_transform(latent_vectors)
        return coords_2d, 'pca'
    else:
        # Use t-SNE for high dimensions
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
        coords_2d = tsne.fit_transform(latent_vectors)
        return coords_2d, 'tsne'


def prepare_variant_comparison(
    original: np.ndarray,
    noisy: np.ndarray,
    reconstructed: np.ndarray,
    n_samples: int = 10
) -> Dict[str, np.ndarray]:
    """Prepare samples for variant comparison visualization.

    Args:
        original: Original images, shape (n_samples, 64)
        noisy: Noisy images (for denoising), shape (n_samples, 64)
        reconstructed: Reconstructed images, shape (n_samples, 64)
        n_samples: Number of samples to return. Default: 10

    Returns:
        Dictionary with 'original', 'noisy', 'reconstructed' samples
    """
    n = min(n_samples, len(original))
    return {
        'original': original[:n],
        'noisy': noisy[:n],
        'reconstructed': reconstructed[:n]
    }


def compute_reconstruction_errors(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> np.ndarray:
    """Compute per-sample reconstruction error (MSE).

    Args:
        original: Original images, shape (n_samples, 64)
        reconstructed: Reconstructed images, shape (n_samples, 64)

    Returns:
        Per-sample MSE, shape (n_samples,)
    """
    return np.mean((original - reconstructed) ** 2, axis=1)


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the MNIST digits dataset.

    Returns:
        Dictionary containing dataset metadata
    """
    return {
        'name': 'MNIST Digits (8x8)',
        'description': 'Handwritten digit images (8x8 grayscale)',
        'n_samples': 1797,
        'n_classes': 10,
        'n_features': 64,
        'image_shape': (8, 8),
        'pixel_range': [0, 1],
        'source': 'sklearn.datasets.load_digits'
    }
