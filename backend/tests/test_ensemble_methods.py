"""Smoke test for Ensemble Methods algorithm."""

from algorithms.ml.ensemble_methods.model import EnsembleMethodsModel
from algorithms.ml.ensemble_methods.data import prepare_data


class TestEnsembleMethodsSmoke:
    def test_train_voting(self):
        """Test voting ensemble."""
        X_train, X_test, y_train, y_test, _ = prepare_data(
            dataset_name='iris',
            test_size=0.2
        )
        model = EnsembleMethodsModel(method='voting')
        result = model.train(X_train, y_train, X_test, y_test)

        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert result['execution_time_ms'] > 0
