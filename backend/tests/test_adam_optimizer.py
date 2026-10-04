"""Smoke tests for Adam Optimizer implementation."""

import pytest
import numpy as np
from algorithms.deep_learning.adam_optimizer import AdamOptimizerModel, AdamOptimizerRequest


class TestAdamOptimizerModel:
    """Test Adam Optimizer model."""

    def test_adam_initialization(self):
        """Test AdamOptimizerModel initialization."""
        request = AdamOptimizerRequest(learning_rate=0.001)
        model = AdamOptimizerModel(request)
        assert model is not None

    def test_adam_request_defaults(self):
        """Test AdamOptimizerRequest with default values."""
        request = AdamOptimizerRequest()
        assert request.learning_rate > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
