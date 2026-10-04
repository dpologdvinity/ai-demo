"""Smoke test for t-SNE algorithm."""

import numpy as np
from algorithms.ml.tsne.model import TSNEModel
from algorithms.ml.tsne.data import load_digits_data


class TestTSNESmoke:
    def test_train_basic(self):
        """Test basic t-SNE training."""
        # Load sample data
        data = load_digits_data()
        X = data['X_train'][:50]  # Use small subset for speed
        y = data['y_train'][:50]

        # Create and fit model with minimal iterations
        model = TSNEModel(n_components=2, n_iter=250, perplexity=10)
        result = model.fit_transform(X, y)

        assert result is not None
        assert 'embedded_data' in result
        assert 'training_time_ms' in result
        assert result['training_time_ms'] > 0
        assert result['n_components'] == 2
