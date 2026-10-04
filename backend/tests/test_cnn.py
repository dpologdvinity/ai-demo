"""Smoke tests for CNN implementation."""

import pytest
import numpy as np
from algorithms.deep_learning.cnn import CNNModel, CNNRequest


class TestCNNModel:
    """Test CNN model."""

    def test_cnn_initialization(self):
        """Test CNNModel initialization."""
        model = CNNModel(
            conv_filters=[16, 32],
            kernel_size=3,
            learning_rate=0.001,
            dropout=0.5
        )
        assert model.model is None  # Model is None until train() is called
        assert isinstance(model.training_history, dict)
        assert 'train_loss' in model.training_history

    def test_cnn_request_defaults(self):
        """Test CNNRequest with default values."""
        request = CNNRequest()
        assert request.epochs > 0
        assert request.batch_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
