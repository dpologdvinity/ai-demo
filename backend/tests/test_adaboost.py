"""Smoke test for AdaBoost algorithm."""

import pytest
from algorithms.ml.adaboost.model import AdaBoostModel
from algorithms.ml.adaboost.data import prepare_data


class TestAdaBoostSmoke:
    def test_train_basic(self):
        """Test basic training."""
        # Prepare data
        X_train, X_test, y_train, y_test, _ = prepare_data(
            dataset_name='iris',
            test_size=0.3,
            random_state=42
        )

        # Train model
        model = AdaBoostModel(n_estimators=10, learning_rate=1.0, algorithm='SAMME.R')
        result = model.train(X_train, y_train, X_test, y_test)

        # Assertions
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert result['execution_time_ms'] > 0
        assert 'predictions' in result
        assert 'learning_curve' in result
        assert len(result['predictions']) == len(y_test)
