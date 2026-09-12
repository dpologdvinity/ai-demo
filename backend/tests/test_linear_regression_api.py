"""Tests for Linear Regression API endpoints."""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestLinearRegressionAPI:
    """Test cases for Linear Regression API endpoints."""

    def test_get_linear_regression_info(self):
        """Test getting Linear Regression algorithm information."""
        response = client.get("/api/ml/linear-regression/info")

        assert response.status_code == 200
        data = response.json()

        assert "metadata" in data
        assert "available_datasets" in data

        metadata = data["metadata"]
        assert metadata["name"] == "Linear Regression"
        assert metadata["slug"] == "linear-regression"
        assert metadata["category"] == "ml"
        assert "parameters" in metadata
        assert "complexity" in metadata

    def test_train_linear_regression_default(self):
        """Test training Linear Regression with default parameters."""
        request_data = {
            "fit_intercept": True,
            "normalize": True,
            "dataset_name": "boston",
            "test_size": 0.3
        }

        response = client.post("/api/ml/linear-regression/train", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "metrics" in data
        assert "predictions" in data
        assert "actual" in data
        assert "visualization_data" in data
        assert "execution_time_ms" in data
        assert "model_info" in data
        assert "parameters_used" in data

        # Check metrics
        metrics = data["metrics"]
        assert "r2_score" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics

        # Check visualization data
        viz_data = data["visualization_data"]
        assert "chart_data" in viz_data
        assert len(viz_data["chart_data"]) > 0

        # Check model info
        model_info = data["model_info"]
        assert "coefficients" in model_info
        assert "intercept" in model_info
        assert "n_features_in" in model_info

    def test_train_linear_regression_no_intercept(self):
        """Test training Linear Regression without intercept."""
        request_data = {
            "fit_intercept": False,
            "normalize": True,
            "dataset_name": "boston",
            "test_size": 0.3
        }

        response = client.post("/api/ml/linear-regression/train", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["model_info"]["intercept"] == 0.0
        assert data["parameters_used"]["fit_intercept"] is False

    def test_train_linear_regression_no_normalization(self):
        """Test training Linear Regression without normalization."""
        request_data = {
            "fit_intercept": True,
            "normalize": False,
            "dataset_name": "boston",
            "test_size": 0.3
        }

        response = client.post("/api/ml/linear-regression/train", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["parameters_used"]["normalize"] is False

    def test_train_linear_regression_custom_test_size(self):
        """Test training Linear Regression with custom test size."""
        request_data = {
            "fit_intercept": True,
            "normalize": True,
            "dataset_name": "boston",
            "test_size": 0.2
        }

        response = client.post("/api/ml/linear-regression/train", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["parameters_used"]["test_size"] == 0.2

    def test_train_linear_regression_invalid_test_size(self):
        """Test training with invalid test size."""
        request_data = {
            "fit_intercept": True,
            "normalize": True,
            "dataset_name": "boston",
            "test_size": 0.05  # Too small
        }

        response = client.post("/api/ml/linear-regression/train", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_train_linear_regression_california_dataset(self):
        """Test training with California Housing dataset."""
        request_data = {
            "fit_intercept": True,
            "normalize": True,
            "dataset_name": "california",
            "test_size": 0.3
        }

        response = client.post("/api/ml/linear-regression/train", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["parameters_used"]["dataset_name"] == "california"

    def test_list_ml_algorithms(self):
        """Test listing all ML algorithms."""
        response = client.get("/api/ml/algorithms")

        assert response.status_code == 200
        algorithms = response.json()

        assert isinstance(algorithms, list)
        assert len(algorithms) > 0

        # Check if Linear Regression is in the list
        lr_algo = next(
            (algo for algo in algorithms if algo["slug"] == "linear-regression"),
            None
        )
        assert lr_algo is not None
        assert lr_algo["name"] == "Linear Regression"
        assert lr_algo["category"] == "ml"
