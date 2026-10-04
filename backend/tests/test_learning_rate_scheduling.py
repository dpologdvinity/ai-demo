"""Smoke tests for Learning Rate Scheduling implementation."""

import pytest
from algorithms.deep_learning.learning_rate_scheduling import LearningRateSchedulingModel, LearningRateSchedulingRequest


class TestLearningRateSchedulingModel:
    """Test Learning Rate Scheduling model."""

    def test_lrs_initialization(self):
        """Test LearningRateSchedulingModel initialization."""
        model = LearningRateSchedulingModel(initial_lr=0.001)
        assert model is not None

    def test_lrs_request_defaults(self):
        """Test LearningRateSchedulingRequest with default values."""
        request = LearningRateSchedulingRequest()
        assert request.initial_lr > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
