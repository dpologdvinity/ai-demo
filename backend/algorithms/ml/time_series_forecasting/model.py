"""Time Series Forecasting model implementations using ARIMA, Prophet, and LSTM."""

import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


if TORCH_AVAILABLE:
    class LSTMForecaster(nn.Module):
        """LSTM neural network for time series forecasting."""

        def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1):
            """Initialize LSTM model.

            Args:
                input_size: Number of input features
                hidden_size: Number of hidden units
                num_layers: Number of LSTM layers
                output_size: Number of output features
            """
            super(LSTMForecaster, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            """Forward pass."""
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
else:
    # Create a placeholder class when torch is not available
    class LSTMForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for LSTM forecasting. Install with: pip install torch")


class TimeSeriesForecastingModel:
    """Time series forecasting model supporting multiple methods."""

    def __init__(self, method: str = 'arima'):
        """Initialize the forecasting model.

        Args:
            method: Forecasting method ('arima', 'prophet', 'lstm', 'compare')
        """
        self.method = method
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    def prepare_lstm_data(self, data: np.ndarray, lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training.

        Args:
            data: Time series data
            lookback: Number of previous time steps to use

        Returns:
            Tuple of (X, y) for training
        """
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)

    def forecast_arima(self, data: np.ndarray, p: int, d: int, q: int, forecast_periods: int,
                      confidence_level: float) -> Dict[str, Any]:
        """Forecast using ARIMA model.

        Args:
            data: Historical time series data
            p: AR order
            d: Differencing order
            q: MA order
            forecast_periods: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals

        Returns:
            Dictionary with forecast results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA forecasting. Install with: pip install statsmodels")

        # Fit ARIMA model
        model = ARIMA(data, order=(p, d, q))
        fitted_model = model.fit()

        # Make forecast
        forecast_result = fitted_model.get_forecast(steps=forecast_periods)
        forecast = forecast_result.predicted_mean

        # Get confidence intervals
        alpha = 1 - confidence_level
        conf_int = forecast_result.conf_int(alpha=alpha)

        # Calculate in-sample predictions for metrics
        in_sample_pred = fitted_model.fittedvalues

        return {
            'forecast': forecast,
            'lower_bound': conf_int.iloc[:, 0].values,
            'upper_bound': conf_int.iloc[:, 1].values,
            'in_sample_pred': in_sample_pred,
            'model': fitted_model,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }

    def forecast_prophet(self, data: np.ndarray, dates: pd.DatetimeIndex, forecast_periods: int,
                        confidence_level: float) -> Dict[str, Any]:
        """Forecast using Facebook Prophet.

        Args:
            data: Historical time series data
            dates: DatetimeIndex for the data
            forecast_periods: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals

        Returns:
            Dictionary with forecast results
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet forecasting. Install with: pip install prophet")

        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': dates,
            'y': data
        })

        # Fit Prophet model
        model = Prophet(
            interval_width=confidence_level,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(df)

        # Make future dataframe
        future = model.make_future_dataframe(periods=forecast_periods, freq=dates.freq)
        forecast_df = model.predict(future)

        # Extract forecast values
        forecast = forecast_df['yhat'].iloc[-forecast_periods:].values
        lower_bound = forecast_df['yhat_lower'].iloc[-forecast_periods:].values
        upper_bound = forecast_df['yhat_upper'].iloc[-forecast_periods:].values

        # In-sample predictions
        in_sample_pred = forecast_df['yhat'].iloc[:len(data)].values

        return {
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'in_sample_pred': in_sample_pred,
            'model': model,
            'forecast_df': forecast_df
        }

    def forecast_lstm(self, data: np.ndarray, forecast_periods: int, lookback: int = 10,
                     epochs: int = 50) -> Dict[str, Any]:
        """Forecast using LSTM neural network.

        Args:
            data: Historical time series data
            forecast_periods: Number of periods to forecast
            lookback: Number of previous time steps to use
            epochs: Number of training epochs

        Returns:
            Dictionary with forecast results
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for LSTM forecasting. Install with: pip install torch")

        # Normalize data
        self.scaler_mean = np.mean(data)
        self.scaler_std = np.std(data)
        data_normalized = (data - self.scaler_mean) / self.scaler_std

        # Prepare data
        X, y = self.prepare_lstm_data(data_normalized, lookback)
        X = torch.FloatTensor(X).unsqueeze(-1)
        y = torch.FloatTensor(y).unsqueeze(-1)

        # Create model
        model = LSTMForecaster(input_size=1, hidden_size=50, num_layers=2, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train model
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Make forecast
        model.eval()
        forecast_normalized = []
        current_sequence = data_normalized[-lookback:].tolist()

        with torch.no_grad():
            for _ in range(forecast_periods):
                input_seq = torch.FloatTensor(current_sequence[-lookback:]).unsqueeze(0).unsqueeze(-1)
                pred = model(input_seq).item()
                forecast_normalized.append(pred)
                current_sequence.append(pred)

        # Denormalize forecast
        forecast = np.array(forecast_normalized) * self.scaler_std + self.scaler_mean

        # Calculate in-sample predictions
        with torch.no_grad():
            in_sample_pred_normalized = model(X).squeeze().numpy()
        in_sample_pred = in_sample_pred_normalized * self.scaler_std + self.scaler_mean

        # Estimate confidence intervals (approximation using training error)
        train_error = np.std(y.numpy() - model(X).detach().numpy())
        margin = 1.96 * train_error * self.scaler_std
        lower_bound = forecast - margin
        upper_bound = forecast + margin

        return {
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'in_sample_pred': in_sample_pred,
            'model': model,
            'train_loss': loss.item()
        }

    def decompose_time_series(self, data: np.ndarray, period: int = 7) -> Dict[str, np.ndarray]:
        """Decompose time series into trend, seasonal, and residual components.

        Args:
            data: Time series data
            period: Period for seasonal decomposition

        Returns:
            Dictionary with decomposition components
        """
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            # Adjust period if data is too short
            if len(data) < 2 * period:
                period = max(2, len(data) // 2)

            decomposition = seasonal_decompose(data, model='additive', period=period, extrapolate_trend='freq')

            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
        except Exception:
            # Return simple decomposition if statsmodels fails
            return {
                'trend': data,
                'seasonal': np.zeros_like(data),
                'residual': np.zeros_like(data)
            }

    def calculate_acf_pacf(self, data: np.ndarray, nlags: int = 20) -> Dict[str, Any]:
        """Calculate ACF and PACF for diagnostics.

        Args:
            data: Time series data
            nlags: Number of lags to compute

        Returns:
            Dictionary with ACF and PACF values
        """
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            from statsmodels.tsa.stattools import acf, pacf

            # Adjust nlags if data is too short
            nlags = min(nlags, len(data) // 2 - 1)

            acf_values = acf(data, nlags=nlags)
            pacf_values = pacf(data, nlags=nlags)

            return {
                'acf': acf_values.tolist(),
                'pacf': pacf_values.tolist(),
                'lags': list(range(nlags + 1))
            }
        except Exception:
            return None

    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast accuracy metrics.

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Dictionary with MAE, RMSE, and MAPE
        """
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual[-min_len:]
        predicted = predicted[-min_len:]

        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # MAPE (avoid division by zero)
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
        else:
            mape = 0.0

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }

    def train(self, data: np.ndarray, dates: pd.DatetimeIndex, forecast_periods: int,
              p: int = 5, d: int = 1, q: int = 0, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Train the forecasting model and make predictions.

        Args:
            data: Historical time series data
            dates: DatetimeIndex for the data
            forecast_periods: Number of periods to forecast ahead
            p: ARIMA AR order
            d: ARIMA differencing order
            q: ARIMA MA order
            confidence_level: Confidence interval level

        Returns:
            Dictionary with forecast results and metrics
        """
        start_time = time.time()

        # Split data into train and test for evaluation
        test_size = min(forecast_periods, len(data) // 5)
        train_data = data[:-test_size]
        test_data = data[-test_size:]

        results = {}

        if self.method == 'arima':
            # ARIMA forecast
            arima_result = self.forecast_arima(train_data, p, d, q, test_size, confidence_level)
            metrics = self.calculate_metrics(test_data, arima_result['forecast'][:test_size])

            # Full forecast on all data
            full_forecast = self.forecast_arima(data, p, d, q, forecast_periods, confidence_level)

            results = {
                'forecast': full_forecast['forecast'].tolist(),
                'lower_bound': full_forecast['lower_bound'].tolist(),
                'upper_bound': full_forecast['upper_bound'].tolist(),
                'metrics': metrics,
                'model_info': {
                    'method': 'ARIMA',
                    'order': f'({p}, {d}, {q})',
                    'aic': full_forecast['aic'],
                    'bic': full_forecast['bic']
                }
            }

        elif self.method == 'prophet':
            # Prophet forecast
            train_dates = dates[:-test_size]
            prophet_result = self.forecast_prophet(train_data, train_dates, test_size, confidence_level)
            metrics = self.calculate_metrics(test_data, prophet_result['forecast'][:test_size])

            # Full forecast on all data
            full_forecast = self.forecast_prophet(data, dates, forecast_periods, confidence_level)

            results = {
                'forecast': full_forecast['forecast'].tolist(),
                'lower_bound': full_forecast['lower_bound'].tolist(),
                'upper_bound': full_forecast['upper_bound'].tolist(),
                'metrics': metrics,
                'model_info': {
                    'method': 'Prophet',
                    'seasonality': 'weekly, yearly'
                }
            }

        elif self.method == 'lstm':
            # LSTM forecast
            lstm_result = self.forecast_lstm(train_data, test_size, lookback=10, epochs=50)
            metrics = self.calculate_metrics(test_data, lstm_result['forecast'][:test_size])

            # Full forecast on all data
            full_forecast = self.forecast_lstm(data, forecast_periods, lookback=10, epochs=50)

            results = {
                'forecast': full_forecast['forecast'].tolist(),
                'lower_bound': full_forecast['lower_bound'].tolist(),
                'upper_bound': full_forecast['upper_bound'].tolist(),
                'metrics': metrics,
                'model_info': {
                    'method': 'LSTM',
                    'architecture': '2 layers, 50 hidden units',
                    'train_loss': full_forecast['train_loss']
                }
            }

        elif self.method == 'compare':
            # Compare all methods
            all_forecasts = {}

            if STATSMODELS_AVAILABLE:
                arima_result = self.forecast_arima(train_data, p, d, q, test_size, confidence_level)
                arima_metrics = self.calculate_metrics(test_data, arima_result['forecast'][:test_size])
                all_forecasts['arima'] = {
                    'metrics': arima_metrics,
                    'forecast': arima_result['forecast'][:test_size].tolist()
                }

            if PROPHET_AVAILABLE:
                train_dates = dates[:-test_size]
                prophet_result = self.forecast_prophet(train_data, train_dates, test_size, confidence_level)
                prophet_metrics = self.calculate_metrics(test_data, prophet_result['forecast'][:test_size])
                all_forecasts['prophet'] = {
                    'metrics': prophet_metrics,
                    'forecast': prophet_result['forecast'][:test_size].tolist()
                }

            if TORCH_AVAILABLE:
                lstm_result = self.forecast_lstm(train_data, test_size, lookback=10, epochs=50)
                lstm_metrics = self.calculate_metrics(test_data, lstm_result['forecast'][:test_size])
                all_forecasts['lstm'] = {
                    'metrics': lstm_metrics,
                    'forecast': lstm_result['forecast'][:test_size].tolist()
                }

            # Use best method for final forecast
            best_method = min(all_forecasts.keys(), key=lambda k: all_forecasts[k]['metrics']['rmse'])

            if best_method == 'arima':
                full_forecast = self.forecast_arima(data, p, d, q, forecast_periods, confidence_level)
            elif best_method == 'prophet':
                full_forecast = self.forecast_prophet(data, dates, forecast_periods, confidence_level)
            else:
                full_forecast = self.forecast_lstm(data, forecast_periods, lookback=10, epochs=50)

            results = {
                'forecast': full_forecast['forecast'].tolist(),
                'lower_bound': full_forecast['lower_bound'].tolist(),
                'upper_bound': full_forecast['upper_bound'].tolist(),
                'metrics': all_forecasts[best_method]['metrics'],
                'model_info': {
                    'method': 'Comparison',
                    'best_method': best_method.upper(),
                    'all_methods': all_forecasts
                }
            }

        # Add decomposition
        decomposition = self.decompose_time_series(data)
        if decomposition:
            results['decomposition'] = {
                'trend': decomposition['trend'].tolist(),
                'seasonal': decomposition['seasonal'].tolist(),
                'residual': decomposition['residual'].tolist()
            }

        # Add ACF/PACF
        acf_pacf = self.calculate_acf_pacf(data)
        if acf_pacf:
            results['acf_data'] = acf_pacf

        execution_time_ms = (time.time() - start_time) * 1000
        results['execution_time_ms'] = execution_time_ms

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            'method': self.method,
            'description': f'{self.method.upper()} time series forecasting model'
        }
