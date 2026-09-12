"""Tests for Linear Regression implementation."""

import pytest
import numpy as np
from algorithms.ml.linear_regression.model import LinearRegressionModel
from algorithms.ml.linear_regression.data import load_linear_regression_data


class TestLinearRegressionModel:
    """Test cases for LinearRegressionModel."""

    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = LinearRegressionModel()
        assert model.fit_intercept is True
        assert model.training_time_ms == 0.0

    def test_initialization_no_intercept(self):
        """Test model initialization without intercept."""
        model = LinearRegressionModel(fit_intercept=False)
        assert model.fit_intercept is False

    def test_train_basic(self):
        """Test basic training functionality."""
        model = LinearRegressionModel()
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([3, 7, 11, 15])

        result = model.train(X_train, y_train)

        assert 'coefficients' in result
        assert 'intercept' in result
        assert 'n_features' in result
        assert 'training_time_ms' in result
        assert result['n_features'] == 2
        assert len(result['coefficients']) == 2
        assert result['training_time_ms'] > 0

    def test_train_empty_data(self):
        """Test training with empty data raises error."""
        model = LinearRegressionModel()
        X_train = np.array([])
        y_train = np.array([])

        with pytest.raises(ValueError, match="X_train cannot be empty"):
            model.train(X_train, y_train)

    def test_train_mismatched_shapes(self):
        """Test training with mismatched shapes raises error."""
        model = LinearRegressionModel()
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="must have same number of samples"):
            model.train(X_train, y_train)

    def test_predict(self):
        """Test prediction functionality."""
        model = LinearRegressionModel()
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([3, 7, 11, 15])

        model.train(X_train, y_train)

        X_test = np.array([[2, 3], [4, 5]])
        predictions = model.predict(X_test)

        assert predictions.shape == (2,)
        assert isinstance(predictions, np.ndarray)

    def test_predict_untrained(self):
        """Test prediction without training raises error."""
        model = LinearRegressionModel()
        X_test = np.array([[1, 2]])

        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(X_test)

    def test_evaluate(self):
        """Test model evaluation."""
        model = LinearRegressionModel()
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([3, 7, 11, 15])

        model.train(X_train, y_train)

        X_test = np.array([[2, 3], [4, 5]])
        y_test = np.array([5, 9])

        metrics = model.evaluate(X_test, y_test)

        assert 'r2_score' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_evaluate_untrained(self):
        """Test evaluation without training raises error."""
        model = LinearRegressionModel()
        X_test = np.array([[1, 2]])
        y_test = np.array([3])

        with pytest.raises(ValueError, match="Model must be trained"):
            model.evaluate(X_test, y_test)

    def test_get_model_info(self):
        """Test getting model information."""
        model = LinearRegressionModel()
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([3, 7, 11, 15])

        model.train(X_train, y_train)
        info = model.get_model_info()

        assert 'fit_intercept' in info
        assert 'coefficients' in info
        assert 'intercept' in info
        assert 'n_features_in' in info
        assert info['fit_intercept'] is True
        assert info['n_features_in'] == 2

    def test_get_model_info_untrained(self):
        """Test getting model info without training raises error."""
        model = LinearRegressionModel()

        with pytest.raises(ValueError, match="Model must be trained"):
            model.get_model_info()


class TestLinearRegressionData:
    """Test cases for data loading utilities."""

    def test_load_data_default(self):
        """Test loading data with default parameters."""
        data = load_linear_regression_data()

        assert 'X_train' in data
        assert 'X_test' in data
        assert 'y_train' in data
        assert 'y_test' in data
        assert 'feature_names' in data
        assert 'description' in data
        assert 'normalized' in data

        assert data['normalized'] is True
        assert data['X_train'].shape[0] > 0
        assert data['X_test'].shape[0] > 0

    def test_load_data_no_normalization(self):
        """Test loading data without normalization."""
        data = load_linear_regression_data(normalize=False)

        assert data['normalized'] is False

    def test_load_data_custom_test_size(self):
        """Test loading data with custom test size."""
        data = load_linear_regression_data(test_size=0.2)

        total_samples = data['X_train'].shape[0] + data['X_test'].shape[0]
        test_ratio = data['X_test'].shape[0] / total_samples

        # Allow some tolerance due to rounding
        assert abs(test_ratio - 0.2) < 0.05

    def test_load_data_invalid_test_size(self):
        """Test loading data with invalid test size raises error."""
        with pytest.raises(ValueError, match="test_size must be between"):
            load_linear_regression_data(test_size=0.05)

        with pytest.raises(ValueError, match="test_size must be between"):
            load_linear_regression_data(test_size=0.6)

    def test_load_data_invalid_dataset(self):
        """Test loading data with invalid dataset name raises error."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            load_linear_regression_data(dataset_name="invalid_dataset")

    def test_load_california_dataset(self):
        """Test loading California Housing dataset explicitly."""
        data = load_linear_regression_data(dataset_name="california")

        assert 'X_train' in data
        assert 'X_test' in data
        assert data['X_train'].shape[1] == 8  # California housing has 8 features
