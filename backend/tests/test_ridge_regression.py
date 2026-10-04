"""Smoke test for Ridge Regression algorithm."""

from algorithms.ml.ridge_regression.model import RidgeRegressionModel
from algorithms.ml.ridge_regression.data import load_housing_data


class TestRidgeRegressionSmoke:
    def test_train_basic(self):
        """Test basic training."""
        # Load data
        data = load_housing_data()
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        # Train model
        model = RidgeRegressionModel(alpha=1.0)
        train_result = model.train(X_train, y_train)

        # Check train result structure
        assert 'coefficients' in train_result
        assert 'intercept' in train_result
        assert 'training_time_ms' in train_result
        assert train_result['training_time_ms'] > 0

        # Evaluate model
        eval_result = model.evaluate(X_test, y_test)
        assert 'r2_score' in eval_result
        assert 'mse' in eval_result
        assert 'rmse' in eval_result
        assert 'mae' in eval_result
