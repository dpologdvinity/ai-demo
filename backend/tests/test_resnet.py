"""Smoke tests for ResNet implementation."""

import pytest
from algorithms.deep_learning.resnet import ResNetModel, ResNetRequest


class TestResNetModel:
    """Test ResNet model."""

    def test_resnet_initialization(self):
        """Test ResNetModel initialization."""
        model = ResNetModel(
            num_classes=10,
            learning_rate=0.001
        )
        assert model.model is not None

    def test_resnet_request_defaults(self):
        """Test ResNetRequest with default values."""
        request = ResNetRequest()
        assert request.num_classes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
