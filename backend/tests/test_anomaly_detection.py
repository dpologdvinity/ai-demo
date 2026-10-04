"""Smoke test for Anomaly Detection algorithm."""

from algorithms.ml.anomaly_detection.model import AnomalyDetectionModel
from algorithms.ml.anomaly_detection.schema import AnomalyDetectionParameters


class TestAnomalyDetectionSmoke:
    def test_train_isolation_forest(self):
        """Test training with Isolation Forest."""
        model = AnomalyDetectionModel()
        params = AnomalyDetectionParameters(
            method='isolation_forest',
            contamination=0.1,
            n_estimators=50,
            dataset='synthetic'
        )
        result = model.train(params)

        assert result.success is True
        assert result.anomaly_info.n_anomalies > 0
        assert result.execution_time_ms > 0
