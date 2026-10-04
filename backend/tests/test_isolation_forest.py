"""Smoke test for Isolation Forest algorithm."""

from algorithms.ml.isolation_forest.model import IsolationForestModel
from algorithms.ml.isolation_forest.schema import IsolationForestParameters


class TestIsolationForestSmoke:
    def test_train_basic(self):
        """Test basic isolation forest."""
        model = IsolationForestModel()
        params = IsolationForestParameters(
            contamination=0.1,
            n_estimators=50
        )
        result = model.train(params)

        assert result.success is True
        assert result.anomaly_info.n_anomalies > 0
        assert result.execution_time_ms > 0
