"""Smoke tests for Feedforward NN implementation."""

import pytest
from algorithms.deep_learning.feedforward_nn import MLPModel, MLPRequest


class TestMLPModel:
    """Test MLP model."""

    def test_mlp_initialization(self):
        """Test MLPModel initialization."""
        model = MLPModel(
            hidden_layers=[64, 32],
            learning_rate=0.001
        )
        assert model.hidden_layers == (64, 32)
        assert model.learning_rate == 0.001
        assert model.activation == 'relu'
        assert model.batch_size == 32
        assert model.model is None

    def test_mlp_request_defaults(self):
        """Test MLPRequest with default values."""
        request = MLPRequest()
        assert request.epochs > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
