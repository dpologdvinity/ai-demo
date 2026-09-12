"""GAN (Generative Adversarial Network) module.

This module implements a GAN for generating digit-like images using PyTorch.
"""

from .model import GANModel, Generator, Discriminator
from .schema import GANRequest, GANResponse
from .data import load_digits_data, get_dataset_info, prepare_data_for_gan

__all__ = [
    'GANModel',
    'Generator',
    'Discriminator',
    'GANRequest',
    'GANResponse',
    'load_digits_data',
    'get_dataset_info',
    'prepare_data_for_gan'
]
