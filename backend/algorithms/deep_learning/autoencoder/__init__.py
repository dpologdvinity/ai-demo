"""Autoencoder module.

This module implements an autoencoder for unsupervised feature learning
and dimensionality reduction using PyTorch.
"""

from .model import AutoencoderModel, Encoder, Decoder
from .schema import AutoencoderRequest, AutoencoderResponse
from .data import (
    load_mnist_data,
    get_dataset_info,
    compute_latent_visualization,
    prepare_sample_comparison
)

__all__ = [
    'AutoencoderModel',
    'Encoder',
    'Decoder',
    'AutoencoderRequest',
    'AutoencoderResponse',
    'load_mnist_data',
    'get_dataset_info',
    'compute_latent_visualization',
    'prepare_sample_comparison'
]
