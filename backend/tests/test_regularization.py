"""Smoke test for Regularization algorithm."""

from algorithms.ml.regularization.model import RegularizationModel
from algorithms.ml.regularization.data import create_overfitting_prone_dataset


class TestRegularizationSmoke:
    def test_l2_regularization(self):
        """Test L2 regularization."""
        # Generate minimal dataset for smoke test
        dataset = create_overfitting_prone_dataset(n_samples=100, n_features=50)

        # Create and train model
        model = RegularizationModel(technique='l2', alpha=0.1, max_iterations=100)
        result = model.train(
            dataset['X_train'],
            dataset['y_train'],
            dataset['X_test'],
            dataset['y_test']
        )

        # Verify results
        assert 'metrics' in result
        assert 'predictions' in result
        assert 'coefficients' in result
        assert result['training_time_ms'] > 0
        assert result['metrics']['test_r2'] is not None
