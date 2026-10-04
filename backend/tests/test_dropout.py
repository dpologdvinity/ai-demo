"""Smoke tests for Dropout implementation."""

import pytest
from algorithms.deep_learning.dropout import DropoutModel, DropoutRequest


class TestDropoutModel:
    """Test Dropout model."""

    def test_dropout_initialization(self):
        """Test DropoutModel initialization."""
        model = DropoutModel(
            input_dim=784,
            learning_rate=0.001
        )
        assert model.model is not None

    def test_dropout_request_defaults(self):
        """Test DropoutRequest with default values."""
        request = DropoutRequest()
        assert request.dropout_rate >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
