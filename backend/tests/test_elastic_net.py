"""Smoke test for Elastic Net algorithm."""

from algorithms.ml.elastic_net.model import ElasticNetModel
from algorithms.ml.elastic_net.data import load_housing_data


class TestElasticNetSmoke:
    def test_train_basic(self):
        """Test basic training."""
        # Load data
        data = load_housing_data()
        X_train, y_train = data['X_train'], data['y_train']

        # Create and train model
        model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)
        result = model.train(X_train, y_train)

        assert result['training_time_ms'] > 0
        assert 'coefficients' in result
        assert 'intercept' in result
        assert result['n_features'] > 0
