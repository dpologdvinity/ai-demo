"""Smoke test for Logistic Regression algorithm."""

from algorithms.ml.logistic_regression.model import LogisticRegressionModel
from algorithms.ml.logistic_regression.data import prepare_data


class TestLogisticRegressionSmoke:
    def test_train_basic(self):
        """Test basic training."""
        X_train, X_test, y_train, y_test, metadata = prepare_data(dataset_name='iris')
        model = LogisticRegressionModel()
        result = model.train(X_train, y_train, X_test, y_test)

        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
