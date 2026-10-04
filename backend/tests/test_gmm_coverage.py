"""Smoke test for Gaussian Mixture Model algorithm."""

import numpy as np
from algorithms.ml.gmm.model import GaussianMixtureModelClass
from algorithms.ml.gmm.data import load_blobs_data


class TestGMMSmoke:
    def test_train_basic(self):
        """Test basic GMM training."""
        model = GaussianMixtureModelClass(n_components=2)
        data = load_blobs_data(n_samples=100, centers=2)
        X = data['X']

        # Fit the model
        fit_result = model.fit(X)
        assert fit_result['converged'] is True
        assert fit_result['training_time_ms'] > 0

        # Evaluate clustering
        metrics = model.evaluate(X)
        assert 'silhouette_score' in metrics
        assert 'bic' in metrics
        assert 'aic' in metrics
