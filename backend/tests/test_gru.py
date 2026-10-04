"""Smoke tests for GRU implementation."""

import pytest
from algorithms.deep_learning.gru import GRUModel, GRURequest


class TestGRUModel:
    """Test GRU model."""

    def test_gru_initialization(self):
        """Test GRUModel initialization."""
        request = GRURequest(hidden_size=32)
        model = GRUModel(request)
        assert model.model is not None

    def test_gru_request_defaults(self):
        """Test GRURequest with default values."""
        request = GRURequest()
        assert request.hidden_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
