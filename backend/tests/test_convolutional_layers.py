"""Smoke tests for Convolutional Layers implementation."""

import pytest
import numpy as np
from algorithms.deep_learning.convolutional_layers import ConvolutionalLayersModel, ConvolutionalLayersRequest


class TestConvolutionalLayersModel:
    """Test Convolutional Layers model."""

    def test_conv_layers_initialization(self):
        """Test ConvolutionalLayersModel initialization."""
        model = ConvolutionalLayersModel()
        assert model is not None

    def test_conv_layers_request_defaults(self):
        """Test ConvolutionalLayersRequest with default values."""
        request = ConvolutionalLayersRequest()
        assert request is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
