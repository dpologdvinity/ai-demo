"""Smoke tests for VAE implementation."""

import pytest
from algorithms.deep_learning.vae import VAEModel, VAERequest


class TestVAEModel:
    """Test VAE model."""

    def test_vae_initialization(self):
        """Test VAEModel initialization."""
        model = VAEModel(
            input_dim=784,
            latent_dim=20,
            learning_rate=0.001
        )
        assert model.model is not None

    def test_vae_request_defaults(self):
        """Test VAERequest with default values."""
        request = VAERequest()
        assert request.epochs > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
