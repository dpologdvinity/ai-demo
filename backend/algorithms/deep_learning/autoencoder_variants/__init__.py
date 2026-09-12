"""Autoencoder Variants module.

This module implements multiple autoencoder variants for comparison:
- Vanilla: Standard encoder-decoder
- Denoising: Trained to remove noise
- Sparse: L1 regularization on activations
- Contractive: Penalty on Jacobian (optional)
"""

from .model import (
    AutoencoderVariantsModel,
    VanillaAutoencoder,
    DenoisingAutoencoder,
    SparseAutoencoder,
    ContractiveAutoencoder
)
from .schema import AutoencoderVariantsRequest, AutoencoderVariantsResponse
from .data import (
    load_mnist_data,
    get_dataset_info,
    compute_latent_visualization,
    prepare_variant_comparison,
    add_noise
)

__all__ = [
    'AutoencoderVariantsModel',
    'VanillaAutoencoder',
    'DenoisingAutoencoder',
    'SparseAutoencoder',
    'ContractiveAutoencoder',
    'AutoencoderVariantsRequest',
    'AutoencoderVariantsResponse',
    'load_mnist_data',
    'get_dataset_info',
    'compute_latent_visualization',
    'prepare_variant_comparison',
    'add_noise'
]
