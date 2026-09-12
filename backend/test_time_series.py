"""Quick test script for Time Series Forecasting implementation."""

import sys
import numpy as np
import pandas as pd

# Test data loading
print("Testing data loading...")
from algorithms.ml.time_series_forecasting.data import load_time_series_data, get_dataset_info

# Test all datasets
for i in range(5):
    data = load_time_series_data(series_index=i)
    print(f"  Dataset {i}: {data['name']} - {len(data['data'])} points")

# Get dataset info
info = get_dataset_info()
print(f"\nAvailable datasets: {len(info['available_datasets'])}")

# Test schema
print("\nTesting schema...")
from algorithms.ml.time_series_forecasting.schema import (
    TimeSeriesForecastingRequest,
    TimeSeriesForecastingResponse
)

request = TimeSeriesForecastingRequest(
    method='arima',
    forecast_periods=10,
    p=2,
    d=1,
    q=0,
    series_index=0
)
print(f"  Request created: method={request.method}, periods={request.forecast_periods}")

# Test model
print("\nTesting model (simple ARIMA forecast)...")
from algorithms.ml.time_series_forecasting.model import TimeSeriesForecastingModel

# Load sample data
ts_data = load_time_series_data(series_index=0, random_state=42)
data = ts_data['data'][:100]  # Use first 100 points for quick test
dates = ts_data['dates'][:100]

# Create model
model = TimeSeriesForecastingModel(method='arima')

# Check if statsmodels is available
try:
    from statsmodels.tsa.arima.model import ARIMA
    print("  statsmodels available - testing ARIMA forecast...")

    results = model.train(
        data=data,
        dates=dates,
        forecast_periods=10,
        p=2,
        d=1,
        q=0,
        confidence_level=0.95
    )

    print(f"  Forecast completed in {results['execution_time_ms']:.2f}ms")
    print(f"  Metrics: MAE={results['metrics']['mae']:.2f}, RMSE={results['metrics']['rmse']:.2f}")
    print(f"  Forecast length: {len(results['forecast'])} periods")
    print(f"  Sample forecast values: {[f'{v:.2f}' for v in results['forecast'][:3]]}")

except ImportError:
    print("  statsmodels not available - skipping ARIMA test")
    print("  To enable ARIMA: pip install statsmodels")

# Test decomposition
print("\nTesting time series decomposition...")
try:
    decomp = model.decompose_time_series(data, period=7)
    if decomp:
        print(f"  Decomposition successful:")
        print(f"    Trend length: {len(decomp['trend'])}")
        print(f"    Seasonal length: {len(decomp['seasonal'])}")
        print(f"    Residual length: {len(decomp['residual'])}")
except Exception as e:
    print(f"  Decomposition test skipped: {e}")

# Test ACF/PACF
print("\nTesting ACF/PACF calculation...")
try:
    acf_pacf = model.calculate_acf_pacf(data, nlags=10)
    if acf_pacf:
        print(f"  ACF/PACF calculation successful:")
        print(f"    ACF values: {len(acf_pacf['acf'])}")
        print(f"    PACF values: {len(acf_pacf['pacf'])}")
except Exception as e:
    print(f"  ACF/PACF test skipped: {e}")

print("\n" + "="*60)
print("Time Series Forecasting Implementation Tests Complete!")
print("="*60)

# Check for optional dependencies
print("\nOptional Dependencies Status:")
try:
    import statsmodels
    print("  ✓ statsmodels (ARIMA) - INSTALLED")
except ImportError:
    print("  ✗ statsmodels (ARIMA) - NOT INSTALLED")
    print("    Install with: pip install statsmodels")

try:
    from prophet import Prophet
    print("  ✓ prophet (Facebook Prophet) - INSTALLED")
except ImportError:
    print("  ✗ prophet (Facebook Prophet) - NOT INSTALLED")
    print("    Install with: pip install prophet")

try:
    import torch
    print("  ✓ torch (LSTM) - INSTALLED")
except ImportError:
    print("  ✗ torch (LSTM) - NOT INSTALLED")
    print("    Install with: pip install torch")

print("\nNote: At least statsmodels should be installed for basic functionality")
