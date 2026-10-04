"""Smoke tests for Batch Normalization implementation."""

import pytest
import numpy as np
from algorithms.deep_learning.batch_normalization import BatchNormModel, BatchNormRequest


class TestBatchNormModel:
    """Test Batch Normalization model."""

    def test_batch_norm_initialization(self):
        """Test BatchNormModel initialization."""
        request = BatchNormRequest()
        model = BatchNormModel(request)
        assert model.request is not None
        assert model.device is not None

    def test_batch_norm_request_defaults(self):
        """Test BatchNormRequest with default values."""
        request = BatchNormRequest()
        assert request.epochs > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
