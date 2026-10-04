"""Smoke test for XGBoost algorithm."""

from algorithms.ml.xgboost.model import XGBoostModel


class TestXGBoostSmoke:
    def test_train_basic(self):
        """Test basic training."""
        model = XGBoostModel()
        result = model.train(
            n_estimators=10,
            max_depth=3,
            dataset_name='iris'
        )

        assert result['success'] is True
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert result['execution_time_ms'] > 0
