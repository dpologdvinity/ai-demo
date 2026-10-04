"""Smoke test for Random Forest algorithm."""

from algorithms.ml.random_forest.model import RandomForestModel
from algorithms.ml.random_forest.schema import RandomForestRequest


class TestRandomForestSmoke:
    def test_train_basic(self):
        """Test basic training."""
        model = RandomForestModel()
        request = RandomForestRequest(
            n_estimators=10,
            max_depth=5
        )
        result = model.train(request)

        assert result.metrics is not None
        assert 'accuracy' in result.metrics
        assert result.metrics['accuracy'] > 0
        assert result.execution_time_ms > 0
        assert len(result.predictions) > 0
        assert result.feature_importance is not None
