"""Smoke tests for Transformer implementation."""

import pytest
from algorithms.deep_learning.transformer import TransformerModel, TransformerRequest


class TestTransformerModel:
    """Test Transformer model."""

    def test_transformer_initialization(self):
        """Test TransformerModel initialization."""
        model = TransformerModel(
            d_model=64,
            nhead=4,
            num_layers=2,
            learning_rate=0.001
        )
        assert model.model is not None

    def test_transformer_request_defaults(self):
        """Test TransformerRequest with default values."""
        request = TransformerRequest()
        assert request.d_model > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
