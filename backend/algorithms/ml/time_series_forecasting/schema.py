"""Pydantic schemas for Time Series Forecasting API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TimeSeriesForecastingRequest(BaseModel):
    """Request schema for time series forecasting.

    Attributes:
        method: Forecasting method to use (default: 'arima', options: ['arima', 'prophet', 'lstm', 'compare'])
        forecast_periods: Number of periods to forecast ahead (default: 30, range: 5-100)
        p: ARIMA AR order (default: 5, range: 0-10)
        d: ARIMA differencing order (default: 1, range: 0-2)
        q: ARIMA MA order (default: 0, range: 0-10)
        series_index: Dataset selector (default: 0, range: 0-4)
        confidence_level: Confidence interval level (default: 0.95, range: 0.8-0.99)
        random_state: Random seed for reproducibility (default: 42)
    """

    method: str = Field(
        default='arima',
        description="Forecasting method to use"
    )
    forecast_periods: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of periods to forecast ahead"
    )
    p: int = Field(
        default=5,
        ge=0,
        le=10,
        description="ARIMA AR (autoregressive) order"
    )
    d: int = Field(
        default=1,
        ge=0,
        le=2,
        description="ARIMA differencing order"
    )
    q: int = Field(
        default=0,
        ge=0,
        le=10,
        description="ARIMA MA (moving average) order"
    )
    series_index: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Index of the time series dataset to use"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.8,
        le=0.99,
        description="Confidence level for prediction intervals"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "method": "arima",
                "forecast_periods": 30,
                "p": 5,
                "d": 1,
                "q": 0,
                "series_index": 0,
                "confidence_level": 0.95,
                "random_state": 42
            }
        }


class TimeSeriesForecastingResponse(BaseModel):
    """Response schema for time series forecasting results.

    Attributes:
        success: Whether the operation completed successfully
        historical_data: Historical time series data points
        forecast: Forecasted values
        lower_bound: Lower confidence interval bound
        upper_bound: Upper confidence interval bound
        metrics: Performance metrics (MAE, RMSE, MAPE)
        decomposition: Time series decomposition (trend, seasonal, residual)
        acf_data: Autocorrelation function data
        pacf_data: Partial autocorrelation function data
        visualization_data: Data formatted for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        model_info: Information about the fitted model
        parameters_used: Parameters that were used for forecasting
        error: Error message if operation failed
    """

    success: bool
    historical_data: Optional[List[float]] = None
    forecast: Optional[List[float]] = None
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    decomposition: Optional[Dict[str, List[float]]] = None
    acf_data: Optional[Dict[str, Any]] = None
    pacf_data: Optional[Dict[str, Any]] = None
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0.0
    model_info: Optional[Dict[str, Any]] = None
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "historical_data": [100.5, 102.3, 98.7, 105.2],
                "forecast": [106.8, 108.1, 107.5],
                "lower_bound": [104.2, 105.1, 104.8],
                "upper_bound": [109.4, 111.1, 110.2],
                "metrics": {
                    "mae": 2.45,
                    "rmse": 3.12,
                    "mape": 2.34
                },
                "decomposition": {
                    "trend": [100.1, 101.2, 102.3],
                    "seasonal": [0.4, 1.1, -0.6],
                    "residual": [0.0, 0.0, 0.0]
                },
                "execution_time_ms": 1250.5,
                "model_info": {
                    "method": "arima",
                    "order": "(5, 1, 0)",
                    "aic": 450.23,
                    "bic": 465.78
                },
                "parameters_used": {
                    "method": "arima",
                    "forecast_periods": 30,
                    "p": 5,
                    "d": 1,
                    "q": 0
                }
            }
        }
