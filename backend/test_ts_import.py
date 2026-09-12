"""Simple import test for Time Series Forecasting module."""

import sys
print("Python:", sys.version)
print()

# Test basic imports
print("Testing module imports...")
try:
    from algorithms.ml.time_series_forecasting import (
        TimeSeriesForecastingModel,
        TimeSeriesForecastingRequest,
        TimeSeriesForecastingResponse
    )
    print("✓ Main classes imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test schema instantiation
print("\nTesting schema instantiation...")
try:
    request = TimeSeriesForecastingRequest(
        method='arima',
        forecast_periods=30,
        p=5,
        d=1,
        q=0,
        series_index=0,
        confidence_level=0.95
    )
    print(f"✓ Request created: {request.method}, {request.forecast_periods} periods")
except Exception as e:
    print(f"✗ Schema instantiation failed: {e}")
    sys.exit(1)

# Test model instantiation
print("\nTesting model instantiation...")
try:
    model = TimeSeriesForecastingModel(method='arima')
    print(f"✓ Model created: {model.method}")

    model_info = model.get_model_info()
    print(f"✓ Model info: {model_info}")
except Exception as e:
    print(f"✗ Model instantiation failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ All basic import and instantiation tests passed!")
print("="*60)
print("\nNote: Full functionality tests require:")
print("  - pandas (for time series handling)")
print("  - statsmodels (for ARIMA)")
print("  - prophet (for Prophet forecasting)")
print("  - torch (for LSTM)")
print("\nInstall with: pip install pandas statsmodels prophet torch")
