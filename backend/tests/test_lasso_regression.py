"""Smoke test for Lasso Regression algorithm."""

import numpy as np
from algorithms.ml.lasso_regression.model import LassoRegressionModel
from algorithms.ml.lasso_regression.data import load_boston_data


class TestLassoRegressionSmoke:
    def test_train_basic(self):
        """Test basic training."""
        # Load data
        data = load_boston_data()
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

        # Train model
        model = LassoRegressionModel(alpha=0.1)
        train_result = model.train(X_train, y_train)

        # Verify train result
        assert isinstance(train_result, dict)
        assert 'coefficients' in train_result
        assert 'intercept' in train_result
        assert train_result['training_time_ms'] > 0
        assert train_result['converged'] is not None

        # Verify predictions work
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

        # Verify evaluation works
        metrics = model.evaluate(X_test, y_test)
        assert 'r2_score' in metrics
        assert 'mse' in metrics
