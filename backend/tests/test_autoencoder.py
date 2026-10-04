"""Smoke tests for Autoencoder implementation."""

import pytest
from algorithms.deep_learning.autoencoder import AutoencoderModel, AutoencoderRequest


class TestAutoencoderModel:
    """Test Autoencoder model."""

    def test_autoencoder_initialization(self):
        """Test AutoencoderModel initialization."""
        model = AutoencoderModel(
            input_dim=784,
            encoding_dim=32,
            learning_rate=0.001
        )
        assert model.model is not None

    def test_autoencoder_request_defaults(self):
        """Test AutoencoderRequest with default values."""
        request = AutoencoderRequest()
        assert request.epochs > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
