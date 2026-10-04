"""Smoke test for Hyperparameter Tuning algorithm."""

from algorithms.ml.hyperparameter_tuning.model import HyperparameterTuningModel
from algorithms.ml.hyperparameter_tuning.schema import HyperparameterTuningRequest


class TestHyperparameterTuningSmoke:
    def test_grid_search_basic(self):
        """Test grid search."""
        model = HyperparameterTuningModel()
        result = model.train(
            method='grid',
            model_type='random_forest',
            dataset='iris'
        )

        assert result['success'] is True
        assert 'best_params' in result
        assert result['execution_time_ms'] > 0
