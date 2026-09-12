"""Data generation and loading for time series forecasting algorithms."""

import numpy as np
from typing import Dict, Any, Tuple

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def generate_stock_prices(n_points: int = 365, random_state: int = 42) -> Tuple[np.ndarray, Any]:
    """Generate synthetic daily stock price data.

    Args:
        n_points: Number of data points to generate
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (prices, dates)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for time series forecasting. Install with: pip install pandas")

    np.random.seed(random_state)

    # Start price
    start_price = 100.0

    # Generate random walk with drift and volatility
    drift = 0.0005  # Small upward drift
    volatility = 0.02
    returns = np.random.normal(drift, volatility, n_points)

    # Convert returns to prices
    prices = start_price * np.exp(np.cumsum(returns))

    # Add some seasonality (weekly pattern)
    for i in range(n_points):
        day_of_week = i % 7
        if day_of_week == 5 or day_of_week == 6:  # Weekend effect
            prices[i] *= 0.998
        elif day_of_week == 0:  # Monday effect
            prices[i] *= 1.002

    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')

    return prices, dates


def generate_temperature_data(n_points: int = 365, random_state: int = 42) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate synthetic monthly temperature data with strong seasonality.

    Args:
        n_points: Number of data points to generate
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (temperatures, dates)
    """
    np.random.seed(random_state)

    # Generate time points
    t = np.linspace(0, n_points / 12, n_points)  # Time in years

    # Base temperature with annual cycle
    base_temp = 15  # Average temperature in Celsius
    amplitude = 15  # Temperature variation

    # Seasonal component (annual cycle)
    seasonal = amplitude * np.sin(2 * np.pi * t)

    # Trend (global warming)
    trend = 0.02 * t

    # Random noise
    noise = np.random.normal(0, 2, n_points)

    # Combine components
    temperatures = base_temp + seasonal + trend + noise

    # Generate dates
    dates = pd.date_range(start='2018-01-01', periods=n_points, freq='D')

    return temperatures, dates


def generate_sales_data(n_points: int = 104, random_state: int = 42) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate synthetic weekly sales data with trend and seasonality.

    Args:
        n_points: Number of data points to generate
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (sales, dates)
    """
    np.random.seed(random_state)

    # Time points
    t = np.arange(n_points)

    # Base sales with trend
    base_sales = 1000
    trend = 5 * t

    # Quarterly seasonality
    seasonal = 200 * np.sin(2 * np.pi * t / 52) + 100 * np.sin(4 * np.pi * t / 52)

    # End-of-year spike (holiday season)
    for i in range(n_points):
        week_of_year = i % 52
        if 48 <= week_of_year <= 51:  # Last weeks of year
            seasonal[i] += 300

    # Random noise
    noise = np.random.normal(0, 100, n_points)

    # Combine components
    sales = base_sales + trend + seasonal + noise
    sales = np.maximum(sales, 0)  # Ensure non-negative

    # Generate dates
    dates = pd.date_range(start='2022-01-03', periods=n_points, freq='W-MON')

    return sales, dates


def generate_web_traffic_data(n_points: int = 365, random_state: int = 42) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate synthetic daily web traffic data with multiple patterns.

    Args:
        n_points: Number of data points to generate
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (traffic, dates)
    """
    np.random.seed(random_state)

    # Time points
    t = np.arange(n_points)

    # Base traffic with exponential growth
    base_traffic = 1000
    growth = base_traffic * np.exp(0.002 * t)

    # Weekly seasonality (lower on weekends)
    weekly_seasonal = np.zeros(n_points)
    for i in range(n_points):
        day_of_week = i % 7
        if day_of_week < 5:  # Weekday
            weekly_seasonal[i] = 200
        else:  # Weekend
            weekly_seasonal[i] = -300

    # Random spikes (viral content)
    spikes = np.zeros(n_points)
    spike_days = np.random.choice(n_points, size=int(n_points * 0.05), replace=False)
    spikes[spike_days] = np.random.uniform(500, 1500, len(spike_days))

    # Random noise
    noise = np.random.normal(0, 150, n_points)

    # Combine components
    traffic = growth + weekly_seasonal + spikes + noise
    traffic = np.maximum(traffic, 0)  # Ensure non-negative

    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')

    return traffic, dates


def generate_sensor_readings(n_points: int = 720, random_state: int = 42) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate synthetic hourly sensor readings with daily patterns.

    Args:
        n_points: Number of data points to generate (hours)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (readings, dates)
    """
    np.random.seed(random_state)

    # Time points
    t = np.arange(n_points)

    # Base reading
    base_reading = 50

    # Daily cycle (temperature/pressure sensor)
    daily_cycle = 10 * np.sin(2 * np.pi * t / 24)

    # Weekly pattern
    weekly_pattern = 5 * np.sin(2 * np.pi * t / (24 * 7))

    # Slow drift
    drift = 0.01 * t

    # Random noise
    noise = np.random.normal(0, 2, n_points)

    # Occasional anomalies
    anomalies = np.zeros(n_points)
    anomaly_indices = np.random.choice(n_points, size=int(n_points * 0.02), replace=False)
    anomalies[anomaly_indices] = np.random.uniform(-15, 15, len(anomaly_indices))

    # Combine components
    readings = base_reading + daily_cycle + weekly_pattern + drift + noise + anomalies

    # Generate dates
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='H')

    return readings, dates


def load_time_series_data(series_index: int = 0, random_state: int = 42) -> Dict[str, Any]:
    """Load a time series dataset by index.

    Args:
        series_index: Index of the dataset (0-4)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing time series data and metadata
    """
    datasets = [
        {
            'name': 'Stock Prices',
            'description': 'Daily stock price data with trend and volatility',
            'generator': generate_stock_prices,
            'frequency': 'Daily',
            'units': 'USD'
        },
        {
            'name': 'Temperature',
            'description': 'Daily temperature with strong seasonal patterns',
            'generator': generate_temperature_data,
            'frequency': 'Daily',
            'units': 'Celsius'
        },
        {
            'name': 'Sales',
            'description': 'Weekly sales data with trend and seasonality',
            'generator': generate_sales_data,
            'frequency': 'Weekly',
            'units': 'Units Sold'
        },
        {
            'name': 'Web Traffic',
            'description': 'Daily web traffic with growth and weekly patterns',
            'generator': generate_web_traffic_data,
            'frequency': 'Daily',
            'units': 'Visits'
        },
        {
            'name': 'Sensor Readings',
            'description': 'Hourly sensor data with daily and weekly cycles',
            'generator': generate_sensor_readings,
            'frequency': 'Hourly',
            'units': 'Measurement Units'
        }
    ]

    if series_index < 0 or series_index >= len(datasets):
        raise ValueError(f"series_index must be between 0 and {len(datasets) - 1}")

    dataset_info = datasets[series_index]
    data, dates = dataset_info['generator'](random_state=random_state)

    return {
        'data': data,
        'dates': dates,
        'name': dataset_info['name'],
        'description': dataset_info['description'],
        'frequency': dataset_info['frequency'],
        'units': dataset_info['units'],
        'n_points': len(data)
    }


def get_dataset_info() -> Dict[str, Any]:
    """Get information about available time series datasets.

    Returns:
        Dictionary with dataset information
    """
    return {
        'available_datasets': [
            {'index': 0, 'name': 'Stock Prices', 'frequency': 'Daily'},
            {'index': 1, 'name': 'Temperature', 'frequency': 'Daily'},
            {'index': 2, 'name': 'Sales', 'frequency': 'Weekly'},
            {'index': 3, 'name': 'Web Traffic', 'frequency': 'Daily'},
            {'index': 4, 'name': 'Sensor Readings', 'frequency': 'Hourly'}
        ],
        'default_index': 0
    }
