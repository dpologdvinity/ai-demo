"""Dropout Regularization demonstration using PyTorch."""

import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .data import generate_overfitting_prone_dataset, create_dropout_mask


class SimpleNetwork(nn.Module):
    """Simple feedforward network with optional dropout.

    This network is designed to demonstrate overfitting and how dropout prevents it.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        dropout_rate: float = 0.0,
        apply_to_layers: List[str] = None
    ):
        """Initialize the network.

        Args:
            input_dim: Number of input features
            hidden_layers: Sizes of hidden layers
            output_dim: Number of output classes
            dropout_rate: Dropout probability
            apply_to_layers: Which layers to apply dropout to
        """
        super(SimpleNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.apply_to_layers = apply_to_layers or []

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Dropout after this layer if specified
            layer_name = f"hidden{i+1}"
            if dropout_rate > 0 and layer_name in self.apply_to_layers:
                layers.append(nn.Dropout(p=dropout_rate))

            prev_dim = hidden_dim

        # Output layer (no dropout here)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output logits
        """
        return self.network(x)


class DropoutModel:
    """Model for demonstrating dropout regularization effects.

    This model trains multiple networks with different dropout rates and
    compares their training/validation performance to show how dropout
    prevents overfitting.
    """

    def __init__(
        self,
        dropout_rate: float = 0.5,
        apply_to_layers: List[str] = None,
        hidden_layers: List[int] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        random_state: int = 42
    ):
        """Initialize dropout demonstration model.

        Args:
            dropout_rate: Primary dropout rate to demonstrate
            apply_to_layers: Which layers get dropout
            hidden_layers: Network architecture
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            random_state: Random seed
        """
        self.dropout_rate = dropout_rate
        self.apply_to_layers = apply_to_layers or ["hidden1", "hidden2"]
        self.hidden_layers = hidden_layers or [128, 64]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.models: Dict[str, SimpleNetwork] = {}
        self.training_histories: Dict[str, Dict[str, List[float]]] = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100
    ) -> Dict[str, Any]:
        """Train networks with different dropout rates to compare.

        Trains three networks:
        1. No dropout (baseline - will overfit)
        2. Light dropout (0.2)
        3. Standard dropout (0.5)
        4. Heavy dropout (0.8)

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs

        Returns:
            Dictionary with training results
        """
        start_time = time.time()

        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_train))

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Dropout rates to compare
        dropout_configs = [
            ("no_dropout", 0.0),
            ("dropout_0.2", 0.2),
            ("dropout_0.5", 0.5),
            ("dropout_0.8", 0.8)
        ]

        # Train each configuration
        for config_name, dropout_rate in dropout_configs:
            print(f"Training with {config_name}...")

            # Create model
            model = SimpleNetwork(
                input_dim=input_dim,
                hidden_layers=self.hidden_layers,
                output_dim=output_dim,
                dropout_rate=dropout_rate,
                apply_to_layers=self.apply_to_layers
            )

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_accuracy': [],
                'val_accuracy': []
            }

            # Training loop
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * batch_X.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()

                train_loss = train_loss / len(train_dataset)
                train_accuracy = train_correct / train_total

                # Validation phase
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_accuracy = (val_predicted == y_val_t).sum().item() / len(y_val_t)

                # Store metrics
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_accuracy'].append(train_accuracy)
                history['val_accuracy'].append(val_accuracy)

            # Store model and history
            self.models[config_name] = model
            self.training_histories[config_name] = history

        training_time = time.time() - start_time

        return {
            'training_time_ms': training_time * 1000,
            'models_trained': len(dropout_configs)
        }

    def get_dropout_masks(self, n_samples: int = 3) -> List[Dict[str, Any]]:
        """Generate example dropout masks for visualization.

        Args:
            n_samples: Number of sample masks to generate

        Returns:
            List of dropout mask examples
        """
        masks = []

        for i in range(n_samples):
            sample_masks = {}

            for layer_idx, layer_size in enumerate(self.hidden_layers):
                layer_name = f"hidden{layer_idx + 1}"

                if layer_name in self.apply_to_layers:
                    # Generate dropout mask
                    mask = create_dropout_mask(
                        shape=(1, layer_size),
                        dropout_rate=self.dropout_rate,
                        random_state=self.random_state + i
                    )[0]  # Get first (and only) sample

                    sample_masks[layer_name] = {
                        'size': layer_size,
                        'mask': mask.tolist(),
                        'active_neurons': int(np.sum(mask > 0)),
                        'dropped_neurons': int(np.sum(mask == 0))
                    }

            masks.append({
                'sample_id': i + 1,
                'dropout_rate': self.dropout_rate,
                'layers': sample_masks
            })

        return masks

    def get_comparison_metrics(self) -> List[Dict[str, Any]]:
        """Get comparison metrics for all trained models.

        Returns:
            List of metrics for each dropout configuration
        """
        comparison = []

        for config_name, history in self.training_histories.items():
            # Get final epoch metrics
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            final_train_acc = history['train_accuracy'][-1]
            final_val_acc = history['val_accuracy'][-1]

            # Calculate overfitting gap
            overfitting_gap = final_train_loss - final_val_loss

            comparison.append({
                'name': config_name,
                'dropout_rate': float(config_name.split('_')[-1]) if 'dropout' in config_name else 0.0,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'final_train_accuracy': final_train_acc,
                'final_val_accuracy': final_val_acc,
                'overfitting_gap': overfitting_gap,
                'generalization_gap': final_train_acc - final_val_acc
            })

        return comparison

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information.

        Returns:
            Dictionary with model details
        """
        # Use the no_dropout model for architecture info
        model = self.models.get('no_dropout')

        if model is None:
            raise ValueError("Model must be trained first")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'architecture': 'feedforward_neural_network',
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'apply_to_layers': self.apply_to_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
