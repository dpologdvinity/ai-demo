"""Smoke test for K-Means algorithm."""

from algorithms.ml.k_means.model import KMeansModel
from algorithms.ml.k_means.schema import KMeansParameters


class TestKMeansSmoke:
    def test_train_basic(self):
        """Test basic K-means training."""
        model = KMeansModel()
        params = KMeansParameters(
            n_clusters=3,
            n_init=10,
            max_iter=300,
            n_samples=300
        )
        result = model.train(params)

        assert result.success is True
        assert result.metrics is not None
        assert 'inertia' in result.metrics
        assert result.execution_time_ms > 0
        assert len(result.clusters) == 3
        assert result.n_iterations > 0
