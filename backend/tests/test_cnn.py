"""Smoke tests for CNN implementation."""

import pytest
import numpy as np
from algorithms.deep_learning.cnn import CNNModel, CNNRequest


class TestCNNModel:
    """Test CNN model."""

    def test_cnn_initialization(self):
        """Test CNNModel initialization."""
        model = CNNModel(
            input_shape=(32, 32, 3),
            num_classes=10,
            learning_rate=0.001
        )
        assert model.model is not None
        assert model.training_history == []

    def test_cnn_request_defaults(self):
        """Test CNNRequest with default values."""
        request = CNNRequest()
        assert request.epochs > 0
        assert request.batch_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
