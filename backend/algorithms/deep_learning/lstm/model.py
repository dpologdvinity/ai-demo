"""LSTM (Long Short-Term Memory) model implementation using PyTorch."""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .data import generate_time_series, prepare_sequences, get_dataset_info
from .schema import (
    LSTMRequest,
    LSTMResponse,
    LSTMMetrics,
    VisualizationData,
    ModelInfo,
    GateActivations
)


class LSTMNetwork(nn.Module):
    """PyTorch LSTM neural network with gate access for visualization."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """Initialize LSTM network.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state and cell state
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability between layers
            output_size: Number of output features
        """
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Store gate activations for visualization
        self.gate_activations = None

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden state tuple (h0, c0)

        Returns:
            Tuple of (output, (hidden_state, cell_state))
        """
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)

        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, hidden)

        # Take the output from the last time step
        out = self.fc(lstm_out[:, -1, :])

        return out, (hn, cn)

    def extract_gate_activations(
        self,
        x: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Extract gate activations for visualization.

        This is a simplified extraction that computes gate activations
        from LSTM weights for a single sample.

        Args:
            x: Input tensor of shape (1, seq_len, input_size)

        Returns:
            Dictionary with gate activations
        """
        self.eval()
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape

            # Initialize hidden states
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

            # Lists to store activations over time
            forget_gates = []
            input_gates = []
            output_gates = []
            cell_states = []
            hidden_states = []

            # Get first layer parameters
            lstm_layer = self.lstm

            # Process each time step
            for t in range(seq_len):
                x_t = x[:, t:t+1, :]  # (batch, 1, input_size)

                # Forward through LSTM for one step
                _, (h_new, c_new) = lstm_layer(x_t, (h, c))

                # Compute gate activations (approximation)
                # For the first layer
                W_ii, W_if, W_ig, W_io = lstm_layer.weight_ih_l0.chunk(4, 0)
                W_hi, W_hf, W_hg, W_ho = lstm_layer.weight_hh_l0.chunk(4, 0)
                b_ii, b_if, b_ig, b_io = lstm_layer.bias_ih_l0.chunk(4, 0)
                b_hi, b_hf, b_hg, b_ho = lstm_layer.bias_hh_l0.chunk(4, 0)

                # Input gate
                i_t = torch.sigmoid(
                    torch.matmul(x_t.squeeze(1), W_ii.t()) +
                    torch.matmul(h[0], W_hi.t()) +
                    b_ii + b_hi
                )
                # Forget gate
                f_t = torch.sigmoid(
                    torch.matmul(x_t.squeeze(1), W_if.t()) +
                    torch.matmul(h[0], W_hf.t()) +
                    b_if + b_hf
                )
                # Output gate
                o_t = torch.sigmoid(
                    torch.matmul(x_t.squeeze(1), W_io.t()) +
                    torch.matmul(h[0], W_ho.t()) +
                    b_io + b_ho
                )

                # Store activations (average across hidden dimensions for visualization)
                forget_gates.append(f_t.mean().item())
                input_gates.append(i_t.mean().item())
                output_gates.append(o_t.mean().item())
                cell_states.append(c_new[0].mean().item())
                hidden_states.append(h_new[0].mean().item())

                # Update hidden states
                h, c = h_new, c_new

            return {
                'forget_gate': forget_gates,
                'input_gate': input_gates,
                'output_gate': output_gates,
                'cell_state': cell_states,
                'hidden_state': hidden_states
            }


class LSTMModel:
    """LSTM model wrapper for training and evaluation."""

    def __init__(self, request: LSTMRequest):
        """Initialize LSTM model.

        Args:
            request: Training parameters
        """
        self.request = request
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize network
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=request.hidden_size,
            num_layers=request.num_layers,
            dropout=request.dropout,
            output_size=1
        ).to(self.device)

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=request.learning_rate
        )

        self.train_losses = []
        self.val_losses = []

    def train(self) -> LSTMResponse:
        """Train the LSTM model.

        Returns:
            LSTMResponse with training results and visualizations
        """
        start_time = time.time()

        try:
            # Generate data
            data = generate_time_series(
                n_samples=self.request.n_samples,
                sequence_length=self.request.sequence_length,
                random_state=self.request.random_state
            )

            # Prepare tensors
            X_train, y_train = prepare_sequences(
                data['X_train'],
                data['y_train'],
                self.device
            )
            X_test, y_test = prepare_sequences(
                data['X_test'],
                data['y_test'],
                self.device
            )

            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.request.batch_size,
                shuffle=True
            )

            # Training loop
            self.train_losses = []
            self.val_losses = []

            for epoch in range(self.request.epochs):
                self.model.train()
                epoch_train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    # Forward pass
                    outputs, _ = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_train_loss += loss.item()

                # Calculate average training loss
                avg_train_loss = epoch_train_loss / len(train_loader)
                self.train_losses.append(avg_train_loss)

                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs, _ = self.model(X_test)
                    val_loss = self.criterion(val_outputs, y_test)
                    self.val_losses.append(val_loss.item())

            # Final evaluation
            metrics = self._evaluate(X_train, y_train, X_test, y_test)

            # Generate visualizations
            visualization_data = self._generate_visualizations(
                X_train, y_train, X_test, y_test, data
            )

            # Model info
            model_info = ModelInfo(
                total_parameters=sum(p.numel() for p in self.model.parameters()),
                hidden_size=self.request.hidden_size,
                num_layers=self.request.num_layers,
                dropout=self.request.dropout,
                sequence_length=self.request.sequence_length,
                device=str(self.device)
            )

            execution_time_ms = (time.time() - start_time) * 1000

            return LSTMResponse(
                success=True,
                metrics=metrics,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                model_info=model_info,
                parameters_used=self.request.model_dump()
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            # Return error response
            return LSTMResponse(
                success=False,
                metrics=LSTMMetrics(
                    train_loss=0.0, test_loss=0.0,
                    train_mse=0.0, test_mse=0.0,
                    train_mae=0.0, test_mae=0.0,
                    train_r2=0.0, test_r2=0.0
                ),
                visualization_data=VisualizationData(
                    training_curves={},
                    predictions=[]
                ),
                execution_time_ms=execution_time_ms,
                model_info=ModelInfo(
                    total_parameters=0,
                    hidden_size=self.request.hidden_size,
                    num_layers=self.request.num_layers,
                    dropout=self.request.dropout,
                    sequence_length=self.request.sequence_length,
                    device=str(self.device)
                ),
                parameters_used=self.request.model_dump(),
                error=str(e)
            )

    def _evaluate(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor
    ) -> LSTMMetrics:
        """Evaluate model performance.

        Args:
            X_train: Training input
            y_train: Training targets
            X_test: Test input
            y_test: Test targets

        Returns:
            LSTMMetrics with performance metrics
        """
        self.model.eval()
        with torch.no_grad():
            # Train predictions
            train_pred, _ = self.model(X_train)
            train_pred_np = train_pred.cpu().numpy()
            y_train_np = y_train.cpu().numpy()

            # Test predictions
            test_pred, _ = self.model(X_test)
            test_pred_np = test_pred.cpu().numpy()
            y_test_np = y_test.cpu().numpy()

            # Calculate metrics
            train_mse = mean_squared_error(y_train_np, train_pred_np)
            test_mse = mean_squared_error(y_test_np, test_pred_np)
            train_mae = mean_absolute_error(y_train_np, train_pred_np)
            test_mae = mean_absolute_error(y_test_np, test_pred_np)
            train_r2 = r2_score(y_train_np, train_pred_np)
            test_r2 = r2_score(y_test_np, test_pred_np)

            return LSTMMetrics(
                train_loss=self.train_losses[-1] if self.train_losses else 0.0,
                test_loss=self.val_losses[-1] if self.val_losses else 0.0,
                train_mse=float(train_mse),
                test_mse=float(test_mse),
                train_mae=float(train_mae),
                test_mae=float(test_mae),
                train_r2=float(train_r2),
                test_r2=float(test_r2)
            )

    def _generate_visualizations(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        data: Dict[str, Any]
    ) -> VisualizationData:
        """Generate visualization data.

        Args:
            X_train: Training input
            y_train: Training targets
            X_test: Test input
            y_test: Test targets
            data: Original dataset

        Returns:
            VisualizationData for frontend visualization
        """
        self.model.eval()
        with torch.no_grad():
            # Predictions on test set
            test_pred, _ = self.model(X_test)
            test_pred_np = test_pred.cpu().numpy().flatten()
            y_test_np = y_test.cpu().numpy().flatten()

            # Prepare predictions vs actual (limit to 100 points for performance)
            predictions = []
            n_viz = min(100, len(test_pred_np))
            for i in range(n_viz):
                predictions.append({
                    'index': i,
                    'predicted': float(test_pred_np[i]),
                    'actual': float(y_test_np[i])
                })

            # Training curves
            training_curves = {
                'epochs': list(range(1, len(self.train_losses) + 1)),
                'train_loss': self.train_losses,
                'val_loss': self.val_losses
            }

            # Extract gate activations for a sample sequence
            sample_idx = 0
            sample_input = X_test[sample_idx:sample_idx+1]  # (1, seq_len, 1)
            gate_acts = self.model.extract_gate_activations(sample_input)

            gate_activations = GateActivations(
                forget_gate=[gate_acts['forget_gate']],
                input_gate=[gate_acts['input_gate']],
                output_gate=[gate_acts['output_gate']],
                cell_state=[gate_acts['cell_state']],
                hidden_state=[gate_acts['hidden_state']],
                time_steps=list(range(len(gate_acts['forget_gate'])))
            )

            # Sequence visualization (input sequence + prediction)
            sequence_visualization = []
            sample_seq = sample_input.cpu().numpy().flatten()
            for i, val in enumerate(sample_seq):
                sequence_visualization.append({
                    'step': i,
                    'value': float(val),
                    'type': 'input'
                })
            sequence_visualization.append({
                'step': len(sample_seq),
                'value': float(test_pred_np[sample_idx]),
                'type': 'prediction'
            })

            return VisualizationData(
                training_curves=training_curves,
                predictions=predictions,
                gate_activations=gate_activations,
                sequence_visualization=sequence_visualization
            )


def train_lstm(request: LSTMRequest) -> LSTMResponse:
    """Train LSTM model with given parameters.

    Args:
        request: Training parameters

    Returns:
        LSTMResponse with training results
    """
    model = LSTMModel(request)
    return model.train()
