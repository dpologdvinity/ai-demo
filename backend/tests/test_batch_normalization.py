"""Smoke tests for Batch Normalization implementation."""

import pytest
import numpy as np
from algorithms.deep_learning.batch_normalization import BatchNormModel, BatchNormRequest


class TestBatchNormModel:
    """Test Batch Normalization model."""

    def test_batch_norm_initialization(self):
        """Test BatchNormModel initialization."""
        model = BatchNormModel(
            input_dim=784,
            learning_rate=0.001
        )
        assert model.model is not None

    def test_batch_norm_request_defaults(self):
        """Test BatchNormRequest with default values."""
        request = BatchNormRequest()
        assert request.epochs > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
