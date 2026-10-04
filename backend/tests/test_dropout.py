"""Smoke tests for Dropout implementation."""

import pytest
from algorithms.deep_learning.dropout import DropoutModel, DropoutRequest


class TestDropoutModel:
    """Test Dropout model."""

    def test_dropout_initialization(self):
        """Test DropoutModel initialization."""
        model = DropoutModel(
            dropout_rate=0.5,
            learning_rate=0.001
        )
        assert model.dropout_rate == 0.5
        assert model.learning_rate == 0.001
        assert model.hidden_layers == [128, 64]
        assert model.apply_to_layers == ["hidden1", "hidden2"]

    def test_dropout_request_defaults(self):
        """Test DropoutRequest with default values."""
        request = DropoutRequest()
        assert request.dropout_rate >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
