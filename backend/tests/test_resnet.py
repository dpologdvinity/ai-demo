"""Smoke tests for ResNet implementation."""

import pytest
from algorithms.deep_learning.resnet import ResNetModel, ResNetRequest


class TestResNetModel:
    """Test ResNet model."""

    def test_resnet_initialization(self):
        """Test ResNetModel initialization."""
        model = ResNetModel(
            model_variant="resnet18",
            use_pretrained=False
        )
        assert model.model is not None

    def test_resnet_request_defaults(self):
        """Test ResNetRequest with default values."""
        request = ResNetRequest()
        assert request.top_k > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
