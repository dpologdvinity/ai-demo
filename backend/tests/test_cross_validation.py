"""Smoke test for Cross Validation algorithm."""

from algorithms.ml.cross_validation.model import train_cross_validation
from algorithms.ml.cross_validation.schema import CrossValidationRequest


class TestCrossValidationSmoke:
    def test_train_basic(self):
        """Test basic cross validation."""
        request = CrossValidationRequest(
            model_type='logistic',
            n_splits=5,
            dataset='iris'
        )
        result = train_cross_validation(request)

        assert result.success is True
        assert result.metrics is not None
        assert result.fold_metrics is not None
        assert len(result.fold_metrics) > 0
        assert result.execution_time_ms > 0
