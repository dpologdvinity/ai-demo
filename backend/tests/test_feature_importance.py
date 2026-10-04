"""Smoke test for Feature Importance algorithm."""

from algorithms.ml.feature_importance.model import FeatureImportanceModel
from algorithms.ml.feature_importance.schema import FeatureImportanceRequest


class TestFeatureImportanceSmoke:
    def test_train_random_forest(self):
        """Test feature importance with random forest."""
        model = FeatureImportanceModel()
        request = FeatureImportanceRequest(
            method='tree',
            model_type='random_forest',
            dataset='housing',
            n_estimators=50
        )
        result = model.train(request)

        assert result.metrics is not None
        assert result.feature_importance is not None
        assert len(result.feature_importance) > 0
        assert result.execution_time_ms > 0
