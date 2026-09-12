"""Time Series Forecasting algorithm implementation.

This package provides time series forecasting using multiple methods:
- ARIMA (Statistical autoregressive integrated moving average)
- Prophet (Facebook's time series forecasting)
- LSTM (Deep learning sequence prediction)
"""

from .model import TimeSeriesForecastingModel
from .schema import TimeSeriesForecastingRequest, TimeSeriesForecastingResponse
from .data import load_time_series_data, get_dataset_info

__all__ = [
    "TimeSeriesForecastingModel",
    "TimeSeriesForecastingRequest",
    "TimeSeriesForecastingResponse",
    "load_time_series_data",
    "get_dataset_info"
]
