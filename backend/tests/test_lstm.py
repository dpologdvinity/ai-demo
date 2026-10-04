"""Smoke tests for LSTM implementation."""

import pytest
from algorithms.deep_learning.lstm import LSTMModel, LSTMRequest


class TestLSTMModel:
    """Test LSTM model."""

    def test_lstm_initialization(self):
        """Test LSTMModel initialization."""
        request = LSTMRequest(
            hidden_size=64,
            num_layers=1,
            learning_rate=0.001
        )
        model = LSTMModel(request)
        assert model.model is not None

    def test_lstm_request_defaults(self):
        """Test LSTMRequest with default values."""
        request = LSTMRequest()
        assert request.hidden_size > 0
        assert request.epochs > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
