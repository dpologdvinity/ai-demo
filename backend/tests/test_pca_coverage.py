"""Smoke test for PCA algorithm."""

from algorithms.ml.pca.model import PCAModel
from algorithms.ml.pca.data import load_digits_data


class TestPCASmoke:
    def test_train_basic(self):
        """Test basic PCA training."""
        model = PCAModel(n_components=2)
        data = load_digits_data()
        X_train = data['X_train']

        result = model.fit_transform(X_train)

        assert isinstance(result, dict)
        assert 'explained_variance_ratio' in result
        assert result['training_time_ms'] >= 0
