"""Tests for RNN (Recurrent Neural Network) implementation."""

import pytest
import numpy as np
import torch
from algorithms.deep_learning.rnn import (
    RNNModel,
    SimpleRNN,
    generate_sine_wave_data,
    TimeSeriesDataset,
    get_dataset_info
)
from torch.utils.data import DataLoader


class TestRNNDataGeneration:
    """Test data generation for RNN."""

    def test_generate_sine_wave_data(self):
        """Test synthetic sine wave data generation."""
        data = generate_sine_wave_data(
            n_samples=1000,
            sequence_length=20,
            prediction_length=10,
            noise_level=0.1,
            random_state=42
        )

        assert "data" in data
        assert "sequences" in data
        assert "targets" in data
        assert "train_indices" in data
        assert "test_indices" in data

        # Check shapes
        assert len(data["data"]) == 1000
        assert data["sequences"].shape[1] == 20
        assert data["targets"].shape[1] == 10

        # Check train/test split
        total_sequences = len(data["sequences"])
        expected_train = int(total_sequences * 0.8)
        assert len(data["train_indices"]) == expected_train

    def test_time_series_dataset(self):
        """Test TimeSeriesDataset creation."""
        sequences = np.random.randn(100, 20)
        targets = np.random.randn(100, 10)

        dataset = TimeSeriesDataset(sequences, targets)

        assert len(dataset) == 100

        seq, target = dataset[0]
        assert seq.shape == (20, 1)
        assert target.shape == (10, 1)
        assert isinstance(seq, torch.Tensor)
        assert isinstance(target, torch.Tensor)

    def test_get_dataset_info(self):
        """Test dataset info retrieval."""
        info = get_dataset_info()

        assert "name" in info
        assert "description" in info
        assert "type" in info
        assert info["features"] == 1
        assert info["temporal"] is True


class TestSimpleRNN:
    """Test SimpleRNN model."""

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = SimpleRNN(input_size=1, hidden_size=64, num_layers=2, output_size=1)

        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert hasattr(model, "rnn")
        assert hasattr(model, "fc")

    def test_forward_pass(self):
        """Test forward pass through the network."""
        model = SimpleRNN(input_size=1, hidden_size=64, num_layers=2, output_size=1)

        # Create dummy input: (batch_size, seq_len, input_size)
        x = torch.randn(8, 20, 1)

        output, hidden = model(x)

        # Output should be (batch_size, output_size)
        assert output.shape == (8, 1)

        # Hidden state should be (num_layers, batch_size, hidden_size)
        assert hidden.shape == (2, 8, 64)

    def test_count_parameters(self):
        """Test parameter counting."""
        model = SimpleRNN(input_size=1, hidden_size=64, num_layers=2, output_size=1)

        param_count = model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)


class TestRNNModel:
    """Test RNNModel wrapper class."""

    def test_model_initialization(self):
        """Test RNNModel initialization."""
        model = RNNModel(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            output_size=1,
            learning_rate=0.001
        )

        assert model.model.hidden_size == 32
        assert model.model.num_layers == 1
        assert model.training_history == []

    def test_training_small_dataset(self):
        """Test training on a small dataset."""
        # Generate small dataset
        data = generate_sine_wave_data(
            n_samples=200,
            sequence_length=10,
            prediction_length=5,
            noise_level=0.05,
            random_state=42
        )

        sequences = data["sequences"][:50]
        targets = data["targets"][:50]

        dataset = TimeSeriesDataset(sequences, targets)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Initialize model
        model = RNNModel(
            input_size=1,
            hidden_size=16,
            num_layers=1,
            output_size=1,
            learning_rate=0.01
        )

        # Train for a few epochs
        training_results = model.train(
            train_loader=loader,
            epochs=5,
            prediction_length=5
        )

        assert "training_history" in training_results
        assert "final_loss" in training_results
        assert "training_time_ms" in training_results
        assert len(training_results["training_history"]) == 5
        assert training_results["training_time_ms"] > 0

    def test_prediction(self):
        """Test making predictions."""
        # Train a simple model first
        data = generate_sine_wave_data(
            n_samples=200,
            sequence_length=10,
            prediction_length=5,
            noise_level=0.05,
            random_state=42
        )

        sequences = data["sequences"][:50]
        targets = data["targets"][:50]

        dataset = TimeSeriesDataset(sequences, targets)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = RNNModel(
            input_size=1,
            hidden_size=16,
            num_layers=1,
            output_size=1,
            learning_rate=0.01
        )

        model.train(train_loader=loader, epochs=3, prediction_length=5)

        # Make predictions on test sequences
        test_sequences = dataset.sequences[:5]
        predictions, hidden_states = model.predict(test_sequences, prediction_length=5)

        assert predictions.shape == (5, 5, 1)
        assert len(hidden_states) == 5

    def test_evaluation(self):
        """Test model evaluation."""
        # Generate data and train
        data = generate_sine_wave_data(
            n_samples=200,
            sequence_length=10,
            prediction_length=5,
            noise_level=0.05,
            random_state=42
        )

        sequences = data["sequences"][:50]
        targets = data["targets"][:50]

        dataset = TimeSeriesDataset(sequences, targets)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = RNNModel(
            input_size=1,
            hidden_size=16,
            num_layers=1,
            output_size=1,
            learning_rate=0.01
        )

        model.train(train_loader=loader, epochs=3, prediction_length=5)

        # Evaluate
        metrics = model.evaluate(test_loader=loader, prediction_length=5)

        assert "loss" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
        assert metrics["loss"] >= 0
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0

    def test_get_model_info(self):
        """Test retrieving model information."""
        model = RNNModel(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            output_size=1,
            learning_rate=0.001
        )

        info = model.get_model_info()

        assert "input_size" in info
        assert "hidden_size" in info
        assert "num_layers" in info
        assert "output_size" in info
        assert "total_parameters" in info
        assert "device" in info

        assert info["hidden_size"] == 32
        assert info["num_layers"] == 2
        assert info["total_parameters"] > 0


class TestRNNSchema:
    """Test RNN request/response schemas."""

    def test_rnn_request_defaults(self):
        """Test RNNRequest with default values."""
        from algorithms.deep_learning.rnn import RNNRequest

        request = RNNRequest()

        assert request.hidden_size == 64
        assert request.num_layers == 2
        assert request.learning_rate == 0.001
        assert request.epochs == 50
        assert request.sequence_length == 20
        assert request.prediction_length == 10

    def test_rnn_request_custom(self):
        """Test RNNRequest with custom values."""
        from algorithms.deep_learning.rnn import RNNRequest

        request = RNNRequest(
            hidden_size=128,
            num_layers=3,
            learning_rate=0.0001,
            epochs=100,
            sequence_length=30,
            prediction_length=15
        )

        assert request.hidden_size == 128
        assert request.num_layers == 3
        assert request.learning_rate == 0.0001
        assert request.epochs == 100
        assert request.sequence_length == 30
        assert request.prediction_length == 15

    def test_rnn_response_structure(self):
        """Test RNNResponse structure."""
        from algorithms.deep_learning.rnn import RNNResponse

        response = RNNResponse(
            success=True,
            metrics={"final_loss": 0.05, "mse": 0.04},
            training_history=[0.5, 0.3, 0.1, 0.05],
            predictions=[[1.0, 1.1, 1.2]],
            actual=[[1.0, 1.1, 1.3]],
            visualization_data={"training_loss": []},
            execution_time_ms=5000.0,
            model_info={"hidden_size": 64},
            parameters_used={"epochs": 50}
        )

        assert response.success is True
        assert response.metrics["final_loss"] == 0.05
        assert len(response.training_history) == 4
        assert response.execution_time_ms == 5000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
