"""Smoke tests for VGG implementation."""

import pytest
from algorithms.deep_learning.vgg import VGGModel, VGGRequest


class TestVGGModel:
    """Test VGG model."""

    def test_vgg_initialization(self):
        """Test VGGModel initialization."""
        model = VGGModel(
            model_variant="vgg16",
            use_pretrained=False
        )
        assert model.model is not None

    def test_vgg_request_defaults(self):
        """Test VGGRequest with default values."""
        request = VGGRequest()
        assert request.top_k > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
