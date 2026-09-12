# Time Series Forecasting Implementation

## Overview

This implementation provides comprehensive time series forecasting capabilities using three different methods:
- **ARIMA** (AutoRegressive Integrated Moving Average) - Statistical method
- **Prophet** (Facebook Prophet) - Business-focused forecasting with seasonality
- **LSTM** (Long Short-Term Memory) - Deep learning approach

## Files Created

### Core Implementation
1. `/backend/algorithms/ml/time_series_forecasting/__init__.py`
   - Package initialization
   - Exports main classes

2. `/backend/algorithms/ml/time_series_forecasting/schema.py`
   - `TimeSeriesForecastingRequest` - API request schema
   - `TimeSeriesForecastingResponse` - API response schema
   - Pydantic models with validation

3. `/backend/algorithms/ml/time_series_forecasting/data.py`
   - 5 diverse synthetic time series datasets:
     * Stock Prices (daily, with trend and volatility)
     * Temperature (daily, strong seasonality)
     * Sales (weekly, trend + seasonality)
     * Web Traffic (daily, growth + weekly patterns)
     * Sensor Readings (hourly, multiple cycles)

4. `/backend/algorithms/ml/time_series_forecasting/model.py`
   - `TimeSeriesForecastingModel` - Main forecasting class
   - `LSTMForecaster` - PyTorch LSTM neural network
   - Methods for ARIMA, Prophet, and LSTM forecasting
   - Time series decomposition (trend, seasonal, residual)
   - ACF/PACF calculation for diagnostics
   - Comprehensive metrics (MAE, RMSE, MAPE)

### API Routes
5. `/backend/api/routes/ml.py` (modified)
   - Added import statements
   - Registered metadata in AlgorithmRegistry
   - Added two endpoints:
     * `POST /ml/time-series-forecasting/train` - Train and forecast
     * `GET /ml/time-series-forecasting/info` - Get algorithm info

## API Endpoints

### Train Time Series Forecasting
```http
POST /api/ml/time-series-forecasting/train
```

**Request Body:**
```json
{
  "method": "arima",           // 'arima', 'prophet', 'lstm', or 'compare'
  "forecast_periods": 30,      // Number of periods to forecast (5-100)
  "p": 5,                      // ARIMA AR order (0-10)
  "d": 1,                      // ARIMA differencing (0-2)
  "q": 0,                      // ARIMA MA order (0-10)
  "series_index": 0,           // Dataset index (0-4)
  "confidence_level": 0.95,    // Confidence interval (0.8-0.99)
  "random_state": 42           // Random seed
}
```

**Response:**
```json
{
  "success": true,
  "historical_data": [100.5, 102.3, ...],
  "forecast": [106.8, 108.1, ...],
  "lower_bound": [104.2, 105.1, ...],
  "upper_bound": [109.4, 111.1, ...],
  "metrics": {
    "mae": 2.45,
    "rmse": 3.12,
    "mape": 2.34
  },
  "decomposition": {
    "trend": [...],
    "seasonal": [...],
    "residual": [...]
  },
  "acf_data": {
    "acf": [...],
    "pacf": [...],
    "lags": [0, 1, 2, ...]
  },
  "visualization_data": {
    "historical": [...],
    "forecast": [...],
    "decomposition": {...}
  },
  "execution_time_ms": 1250.5,
  "model_info": {
    "method": "ARIMA",
    "order": "(5, 1, 0)",
    "aic": 450.23,
    "bic": 465.78
  },
  "parameters_used": {...}
}
```

### Get Algorithm Info
```http
GET /api/ml/time-series-forecasting/info
```

Returns algorithm metadata, available datasets, and method information.

## Features

### Multiple Forecasting Methods

#### 1. ARIMA
- Statistical method for stationary time series
- Parameters: (p, d, q) order
- Provides AIC/BIC for model selection
- Good for linear patterns

#### 2. Prophet
- Facebook's forecasting algorithm
- Automatically detects seasonality
- Robust to missing data and outliers
- Best for business time series

#### 3. LSTM
- Deep learning recurrent neural network
- Learns complex non-linear patterns
- 2-layer architecture with 50 hidden units
- Best for complex dependencies

#### 4. Compare Mode
- Runs all available methods
- Selects best performer based on RMSE
- Returns comparison metrics

### Time Series Decomposition
- Separates into trend, seasonal, and residual components
- Uses additive or multiplicative models
- Helps understand underlying patterns

### Diagnostic Tools
- **ACF (Autocorrelation Function)**: Shows correlation with lagged values
- **PACF (Partial Autocorrelation Function)**: Shows direct correlation
- Helps determine ARIMA orders

### Confidence Intervals
- Prediction intervals for uncertainty quantification
- Adjustable confidence level (80%-99%)
- Different calculation methods per algorithm

### Performance Metrics
- **MAE** (Mean Absolute Error): Average absolute difference
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error

## Datasets

### 0. Stock Prices (Daily)
- 365 data points
- Random walk with drift
- Weekly seasonality
- Volatility patterns

### 1. Temperature (Daily)
- 365 data points
- Strong annual seasonality
- Small upward trend
- Weather-like patterns

### 2. Sales (Weekly)
- 104 data points (2 years)
- Linear growth trend
- Quarterly seasonality
- Holiday spikes

### 3. Web Traffic (Daily)
- 365 data points
- Exponential growth
- Weekly patterns (weekday/weekend)
- Random viral spikes

### 4. Sensor Readings (Hourly)
- 720 data points (30 days)
- Daily cycles
- Weekly patterns
- Occasional anomalies

## Dependencies

### Required
- `numpy` - Array operations
- `pandas` - Time series handling
- `scikit-learn` - ML utilities
- `pydantic` - Schema validation
- `fastapi` - API framework

### Optional (for specific methods)
- `statsmodels>=0.14.4` - ARIMA forecasting
- `prophet>=1.1.5` - Prophet forecasting
- `torch>=2.4.1` - LSTM forecasting

### Installation
```bash
# Core dependencies (should already be installed)
pip install numpy pandas scikit-learn

# Time series specific
pip install statsmodels prophet torch
```

## Usage Examples

### Python Client
```python
import requests

# ARIMA forecast
response = requests.post('http://localhost:8000/api/ml/time-series-forecasting/train', json={
    'method': 'arima',
    'forecast_periods': 30,
    'p': 5,
    'd': 1,
    'q': 0,
    'series_index': 0,
    'confidence_level': 0.95
})

result = response.json()
print(f"MAE: {result['metrics']['mae']:.2f}")
print(f"Forecast: {result['forecast'][:5]}")
```

### Compare Methods
```python
# Compare all methods
response = requests.post('http://localhost:8000/api/ml/time-series-forecasting/train', json={
    'method': 'compare',
    'forecast_periods': 30,
    'series_index': 1  # Temperature data
})

result = response.json()
best_method = result['model_info']['best_method']
print(f"Best method: {best_method}")
```

## Visualization Components

The frontend should implement the following visualizations:

### 1. Main Forecast Chart (Line Chart)
- X-axis: Time (dates/indices)
- Y-axis: Values
- Series:
  - Historical data (solid line)
  - Forecast (dashed line, different color)
  - Confidence interval (shaded area)

### 2. Decomposition Plots (3 Subplots)
- Trend component
- Seasonal component
- Residual component
- All aligned vertically with shared x-axis

### 3. ACF/PACF Correlograms (2 Bar Charts)
- Autocorrelation values
- Confidence bounds
- Lag on x-axis

### 4. Metrics Display
- MAE, RMSE, MAPE cards
- Method comparison table (if compare mode)

### 5. Interactive Controls
- Forecast horizon slider
- Method selector
- Dataset selector
- Parameter adjustments for ARIMA (p, d, q)
- Confidence level slider

## Algorithm Metadata

Registered in `AlgorithmRegistry` with:
- **ID**: `time-series-forecasting`
- **Name**: Time Series Forecasting
- **Category**: ML
- **Difficulty**: Advanced
- **Tags**: ml, time-series, forecasting, arima, prophet, lstm, temporal

## Error Handling

The implementation includes comprehensive error handling:
- Missing dependencies (statsmodels, prophet, torch)
- Invalid parameters (negative periods, invalid orders)
- Data validation errors
- Model fitting failures
- Graceful degradation when optional dependencies missing

## Performance Considerations

- ARIMA: Fast for small datasets (<1000 points)
- Prophet: Moderate speed, handles larger datasets well
- LSTM: Slower training, requires more data
- Compare mode: Runs multiple methods, takes longer

## Testing

Basic syntax validation:
```bash
cd backend
python -m py_compile algorithms/ml/time_series_forecasting/*.py
python -c "import ast; ast.parse(open('api/routes/ml.py').read())"
```

Full testing requires installed dependencies:
```bash
cd backend
pip install statsmodels prophet torch
python test_time_series.py
```

## Future Enhancements

Potential improvements:
1. Add seasonal ARIMA (SARIMA)
2. Add exponential smoothing (Holt-Winters)
3. Add auto-ARIMA for automatic parameter selection
4. Add multivariate forecasting
5. Add real-world datasets (via file upload)
6. Add model persistence (save/load trained models)
7. Add cross-validation for better metric estimation
8. Add residual diagnostics (Ljung-Box test)
9. Add feature engineering (lag features, rolling stats)
10. Add ensemble methods combining multiple forecasters

## Notes

- The implementation prioritizes educational value and demonstration
- All three methods can work independently if dependencies available
- Synthetic datasets ensure reproducible demonstrations
- Confidence intervals use different methods per algorithm:
  - ARIMA: Statistical prediction intervals
  - Prophet: Uncertainty sampling
  - LSTM: Training error approximation
- The compare mode helps users understand which method works best for their data characteristics
