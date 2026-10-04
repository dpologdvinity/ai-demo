"""Smoke test for K-Nearest Neighbors algorithm."""

from algorithms.ml.knn.model import train_knn
from algorithms.ml.knn.schema import KNNRequest


class TestKNNSmoke:
    def test_train_basic(self):
        """Test basic KNN training."""
        request = KNNRequest(
            n_neighbors=5,
            test_size=0.2
        )
        result = train_knn(request)

        assert result.success is True
        assert result.metrics.accuracy > 0
        assert result.execution_time_ms > 0
