"""Smoke tests for Autoencoder implementation."""

import pytest
from algorithms.deep_learning.autoencoder import AutoencoderModel, AutoencoderRequest


class TestAutoencoderModel:
    """Test Autoencoder model."""

    def test_autoencoder_initialization(self):
        """Test AutoencoderModel initialization."""
        model = AutoencoderModel(
            latent_dim=32,
            hidden_dim=128,
            learning_rate=0.001
        )
        assert model.encoder is not None
        assert model.decoder is not None

    def test_autoencoder_request_defaults(self):
        """Test AutoencoderRequest with default values."""
        request = AutoencoderRequest()
        assert request.epochs > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
