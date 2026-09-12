"""RNN model implementation using PyTorch.

This module provides a SimpleRNN class for time series prediction using
recurrent neural networks with PyTorch.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time


class SimpleRNN(nn.Module):
    """Simple RNN architecture for time series prediction.

    This model uses PyTorch's RNN layer followed by a linear layer
    to predict future values in a time series.

    Attributes:
        rnn: PyTorch RNN layer
        fc: Fully connected output layer
        hidden_size: Size of hidden state
        num_layers: Number of RNN layers
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1
    ):
        """Initialize the RNN model.

        Args:
            input_size: Number of input features (1 for univariate time series)
            hidden_size: Dimension of hidden state
            num_layers: Number of stacked RNN layers
            output_size: Number of output features
        """
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Initial hidden state (optional)

        Returns:
            Tuple of (output, hidden_state)
        """
        # RNN layer
        out, hidden = self.rnn(x, hidden)

        # Take the last time step
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        return out, hidden

    def count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RNNModel:
    """Wrapper class for training and evaluating RNN models.

    This class provides a high-level interface for training RNN models
    on time series data, including training, prediction, and evaluation.

    Attributes:
        model: PyTorch RNN model
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run computations on (cpu or cuda)
        training_history: List of loss values during training
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.001
    ):
        """Initialize RNN model wrapper.

        Args:
            input_size: Number of input features
            hidden_size: Hidden state dimension
            num_layers: Number of RNN layers
            output_size: Number of output features
            learning_rate: Learning rate for optimizer
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SimpleRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.training_history = []
        self.training_time_ms = 0.0

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 50,
        prediction_length: int = 10
    ) -> Dict[str, Any]:
        """Train the RNN model.

        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
            prediction_length: Number of steps to predict ahead

        Returns:
            Dictionary containing training metrics and history
        """
        start_time = time.time()
        self.model.train()
        self.training_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Forward pass - predict one step at a time
                self.optimizer.zero_grad()
                predictions = []
                current_seq = sequences

                for step in range(prediction_length):
                    pred, _ = self.model(current_seq)
                    predictions.append(pred)

                    # Update sequence for next prediction
                    current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)

                # Stack predictions
                predictions = torch.stack(predictions, dim=1)

                # Calculate loss
                loss = self.criterion(predictions, targets)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Average loss for the epoch
            avg_loss = epoch_loss / n_batches
            self.training_history.append(avg_loss)

        self.training_time_ms = (time.time() - start_time) * 1000

        return {
            "training_history": self.training_history,
            "final_loss": self.training_history[-1] if self.training_history else 0.0,
            "training_time_ms": self.training_time_ms
        }

    def predict(
        self,
        sequences: torch.Tensor,
        prediction_length: int = 10
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Make predictions on sequences.

        Args:
            sequences: Input sequences, shape (batch, seq_len, features)
            prediction_length: Number of steps to predict ahead

        Returns:
            Tuple of (predictions, hidden_states)
                - predictions: Predicted values, shape (batch, prediction_length, features)
                - hidden_states: List of hidden states for visualization
        """
        self.model.eval()
        sequences = sequences.to(self.device)

        with torch.no_grad():
            predictions = []
            hidden_states = []
            current_seq = sequences

            for step in range(prediction_length):
                pred, hidden = self.model(current_seq)
                predictions.append(pred)
                hidden_states.append(hidden.cpu())

                # Update sequence for next prediction
                current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)

            predictions = torch.stack(predictions, dim=1)

        return predictions, hidden_states

    def evaluate(
        self,
        test_loader: DataLoader,
        prediction_length: int = 10
    ) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            test_loader: DataLoader for test data
            prediction_length: Number of steps to predict ahead

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        n_batches = 0

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Predict
                predictions, _ = self.predict(sequences, prediction_length)

                # Calculate metrics
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()

                # MSE and MAE
                mse = torch.mean((predictions - targets) ** 2).item()
                mae = torch.mean(torch.abs(predictions - targets)).item()

                total_mse += mse
                total_mae += mae
                n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "mae": total_mae / n_batches
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information.

        Returns:
            Dictionary containing model details
        """
        return {
            "input_size": 1,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
            "output_size": 1,
            "total_parameters": self.model.count_parameters(),
            "device": str(self.device)
        }
