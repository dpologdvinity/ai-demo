"""Smoke test for Time Series Forecasting algorithm."""

import pytest

try:
    import pandas as pd
    import numpy as np
    from algorithms.ml.time_series_forecasting.model import TimeSeriesForecastingModel
    PANDAS_AVAILABLE = True
except (ImportError, AttributeError):
    PANDAS_AVAILABLE = False


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas and statsmodels required")
class TestTimeSeriesForecastingSmoke:
    def test_model_instantiation(self):
        """Test model can be instantiated with different methods."""
        for method in ['arima', 'prophet', 'lstm', 'compare']:
            model = TimeSeriesForecastingModel(method=method)
            assert model.method == method
            assert model.model is None

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas required")
    def test_metrics_calculation(self):
        """Test forecast metrics calculation."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        model = TimeSeriesForecastingModel(method='arima')
        metrics = model.calculate_metrics(actual, predicted)

        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
