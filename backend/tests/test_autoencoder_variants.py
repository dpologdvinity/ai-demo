"""Smoke tests for Autoencoder Variants implementation."""

import pytest
from algorithms.deep_learning.autoencoder_variants import AutoencoderVariantsModel, AutoencoderVariantsRequest


class TestAutoencoderVariantsModel:
    """Test Autoencoder Variants model."""

    def test_autoencoder_variants_initialization(self):
        """Test AutoencoderVariantsModel initialization."""
        model = AutoencoderVariantsModel(
            input_dim=784,
            encoding_dim=32,
            variant='sparse'
        )
        assert model is not None

    def test_autoencoder_variants_request_defaults(self):
        """Test AutoencoderVariantsRequest with default values."""
        request = AutoencoderVariantsRequest()
        assert request.epochs > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
