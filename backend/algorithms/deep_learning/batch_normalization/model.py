"""Batch Normalization demonstration using PyTorch.

This module implements two neural networks (with and without batch normalization)
to demonstrate the benefits of BN on training stability, convergence speed,
and gradient flow.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import generate_training_data
from .schema import BatchNormRequest


class DeepNetworkWithBN(nn.Module):
    """Deep neural network with Batch Normalization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        """Initialize network with batch normalization.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_classes: Number of output classes
            momentum: Momentum for running stats
            eps: Epsilon for numerical stability
            affine: Whether to use learnable parameters
            track_running_stats: Whether to track running statistics
        """
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=momentum, eps=eps,
                          affine=affine, track_running_stats=track_running_stats),
            nn.ReLU(),

            # Layer 2
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=momentum, eps=eps,
                          affine=affine, track_running_stats=track_running_stats),
            nn.ReLU(),

            # Layer 3
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=momentum, eps=eps,
                          affine=affine, track_running_stats=track_running_stats),
            nn.ReLU(),

            # Output layer
            nn.Linear(hidden_size, num_classes)
        )

        # Store BN layers for analysis
        self.bn_layers = [m for m in self.modules() if isinstance(m, nn.BatchNorm1d)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def get_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate activations for visualization."""
        activations = []
        with torch.no_grad():
            for layer in self.network:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    activations.append(x.clone())
        return activations


class DeepNetworkWithoutBN(nn.Module):
    """Deep neural network without Batch Normalization."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        """Initialize network without batch normalization.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_classes: Number of output classes
        """
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            # Layer 2
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            # Layer 3
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            # Output layer
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def get_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate activations for visualization."""
        activations = []
        with torch.no_grad():
            for layer in self.network:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    activations.append(x.clone())
        return activations


class BatchNormModel:
    """Wrapper for training and comparing models with/without BN."""

    def __init__(self, request: BatchNormRequest):
        """Initialize the model wrapper.

        Args:
            request: Request containing training parameters
        """
        self.request = request
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Models will be initialized in train()
        self.model_with_bn: Optional[DeepNetworkWithBN] = None
        self.model_without_bn: Optional[DeepNetworkWithoutBN] = None

        # Training history
        self.history_with_bn: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': [],
            'gradient_norms': []
        }
        self.history_without_bn: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': [],
            'gradient_norms': []
        }

        # Activation statistics
        self.activation_stats_with_bn: List[Dict[str, Any]] = []
        self.activation_stats_without_bn: List[Dict[str, Any]] = []

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        track_stats: bool = True
    ) -> Tuple[List[float], List[float], List[float], List[Dict[str, Any]]]:
        """Train a single model and collect statistics.

        Args:
            model: Neural network model
            train_loader: Training data loader
            test_loader: Test data loader
            optimizer: Optimizer
            criterion: Loss function
            epochs: Number of training epochs
            track_stats: Whether to track activation statistics

        Returns:
            Tuple of (loss_history, accuracy_history, gradient_norms, activation_stats)
        """
        loss_history = []
        accuracy_history = []
        gradient_norms = []
        activation_stats = []

        model.to(self.device)

        # Sample batch for activation analysis
        sample_batch = next(iter(test_loader))[0].to(self.device)

        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Track gradient norms
                if track_stats:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    gradient_norms.append(total_norm)

                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                epoch_correct += (predicted == batch_y).sum().item()
                epoch_total += batch_y.size(0)

            # Record training metrics
            avg_loss = epoch_loss / len(train_loader)
            accuracy = epoch_correct / epoch_total
            loss_history.append(avg_loss)
            accuracy_history.append(accuracy)

            # Track activation statistics every 5 epochs
            if track_stats and (epoch + 1) % 5 == 0:
                model.eval()
                activations = model.get_activations(sample_batch)
                stats = {
                    'epoch': epoch + 1,
                    'layers': []
                }
                for layer_idx, act in enumerate(activations):
                    act_np = act.cpu().numpy()
                    stats['layers'].append({
                        'layer': layer_idx + 1,
                        'mean': float(np.mean(act_np)),
                        'std': float(np.std(act_np)),
                        'min': float(np.min(act_np)),
                        'max': float(np.max(act_np))
                    })
                activation_stats.append(stats)

        return loss_history, accuracy_history, gradient_norms, activation_stats

    def train(self) -> Dict[str, Any]:
        """Train both models and collect comparison data.

        Returns:
            Dictionary with training results and comparisons
        """
        start_time = time.time()

        # Generate data
        data = generate_training_data(
            n_samples=1000,
            n_features=20,
            n_classes=10,
            random_state=self.request.random_state
        )

        train_loader = DataLoader(
            data['train_dataset'],
            batch_size=self.request.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            data['test_dataset'],
            batch_size=self.request.batch_size,
            shuffle=False
        )

        # Initialize models
        self.model_with_bn = DeepNetworkWithBN(
            input_size=data['n_features'],
            hidden_size=self.request.hidden_size,
            num_classes=data['n_classes'],
            momentum=self.request.momentum,
            eps=self.request.eps,
            affine=self.request.affine,
            track_running_stats=self.request.track_running_stats
        )

        self.model_without_bn = DeepNetworkWithoutBN(
            input_size=data['n_features'],
            hidden_size=self.request.hidden_size,
            num_classes=data['n_classes']
        )

        # Set up training
        criterion = nn.CrossEntropyLoss()
        optimizer_with_bn = optim.SGD(
            self.model_with_bn.parameters(),
            lr=self.request.learning_rate
        )
        optimizer_without_bn = optim.SGD(
            self.model_without_bn.parameters(),
            lr=self.request.learning_rate
        )

        # Train model WITH batch normalization
        (loss_with_bn, acc_with_bn, grad_with_bn, act_stats_with_bn) = self.train_model(
            self.model_with_bn,
            train_loader,
            test_loader,
            optimizer_with_bn,
            criterion,
            self.request.epochs,
            track_stats=True
        )

        # Train model WITHOUT batch normalization
        (loss_without_bn, acc_without_bn, grad_without_bn, act_stats_without_bn) = self.train_model(
            self.model_without_bn,
            train_loader,
            test_loader,
            optimizer_without_bn,
            criterion,
            self.request.epochs,
            track_stats=True
        )

        # Store results
        self.history_with_bn = {
            'loss': loss_with_bn,
            'accuracy': acc_with_bn,
            'gradient_norms': grad_with_bn
        }
        self.history_without_bn = {
            'loss': loss_without_bn,
            'accuracy': acc_without_bn,
            'gradient_norms': grad_without_bn
        }
        self.activation_stats_with_bn = act_stats_with_bn
        self.activation_stats_without_bn = act_stats_without_bn

        execution_time_ms = (time.time() - start_time) * 1000

        # Analyze convergence
        convergence_threshold = 0.5
        epochs_to_converge_with_bn = self._find_convergence_epoch(
            loss_with_bn, convergence_threshold
        )
        epochs_to_converge_without_bn = self._find_convergence_epoch(
            loss_without_bn, convergence_threshold
        )

        # Prepare response
        return self._prepare_response(
            execution_time_ms,
            epochs_to_converge_with_bn,
            epochs_to_converge_without_bn,
            data['n_features'],
            data['n_classes']
        )

    def _find_convergence_epoch(
        self,
        loss_history: List[float],
        threshold: float
    ) -> Optional[int]:
        """Find the epoch where loss first drops below threshold.

        Args:
            loss_history: List of loss values per epoch
            threshold: Convergence threshold

        Returns:
            Epoch number (1-indexed) or None if never converged
        """
        for epoch, loss in enumerate(loss_history):
            if loss < threshold:
                return epoch + 1
        return None

    def _prepare_response(
        self,
        execution_time_ms: float,
        epochs_to_converge_with_bn: Optional[int],
        epochs_to_converge_without_bn: Optional[int],
        n_features: int,
        n_classes: int
    ) -> Dict[str, Any]:
        """Prepare the response dictionary.

        Args:
            execution_time_ms: Total execution time
            epochs_to_converge_with_bn: Convergence epoch for BN model
            epochs_to_converge_without_bn: Convergence epoch for non-BN model
            n_features: Number of input features
            n_classes: Number of output classes

        Returns:
            Dictionary formatted for API response
        """
        final_loss_with_bn = self.history_with_bn['loss'][-1]
        final_loss_without_bn = self.history_without_bn['loss'][-1]

        improvement_percent = (
            (final_loss_without_bn - final_loss_with_bn) / final_loss_without_bn * 100
        )

        # Metrics
        metrics = {
            'with_bn_final_loss': float(final_loss_with_bn),
            'without_bn_final_loss': float(final_loss_without_bn),
            'with_bn_final_accuracy': float(self.history_with_bn['accuracy'][-1]),
            'without_bn_final_accuracy': float(self.history_without_bn['accuracy'][-1]),
            'improvement_percent': float(improvement_percent),
            'with_bn_epochs_to_converge': epochs_to_converge_with_bn,
            'without_bn_epochs_to_converge': epochs_to_converge_without_bn
        }

        # Training curves
        epochs_list = list(range(1, self.request.epochs + 1))
        training_curves = {
            'epochs': epochs_list,
            'with_bn_loss': [float(x) for x in self.history_with_bn['loss']],
            'without_bn_loss': [float(x) for x in self.history_without_bn['loss']],
            'with_bn_accuracy': [float(x) for x in self.history_with_bn['accuracy']],
            'without_bn_accuracy': [float(x) for x in self.history_without_bn['accuracy']]
        }

        # Convergence comparison
        convergence_comparison = {
            'threshold': 0.5,
            'with_bn_converged': epochs_to_converge_with_bn is not None,
            'without_bn_converged': epochs_to_converge_without_bn is not None,
            'with_bn_epochs': epochs_to_converge_with_bn,
            'without_bn_epochs': epochs_to_converge_without_bn,
            'speedup_factor': (
                epochs_to_converge_without_bn / epochs_to_converge_with_bn
                if epochs_to_converge_with_bn and epochs_to_converge_without_bn
                else None
            )
        }

        # Gradient flow comparison
        gradient_flow = {
            'with_bn_mean_gradient': float(np.mean(self.history_with_bn['gradient_norms'])),
            'without_bn_mean_gradient': float(np.mean(self.history_without_bn['gradient_norms'])),
            'with_bn_gradient_std': float(np.std(self.history_with_bn['gradient_norms'])),
            'without_bn_gradient_std': float(np.std(self.history_without_bn['gradient_norms']))
        }

        # Activation distributions
        activation_distributions = {
            'with_bn': self.activation_stats_with_bn,
            'without_bn': self.activation_stats_without_bn
        }

        # Model info
        model_info = {
            'input_size': n_features,
            'hidden_size': self.request.hidden_size,
            'num_classes': n_classes,
            'num_layers': 3,
            'bn_parameters': {
                'momentum': self.request.momentum,
                'eps': self.request.eps,
                'affine': self.request.affine,
                'track_running_stats': self.request.track_running_stats
            }
        }

        # Visualization data
        visualization_data = {
            'training_curves': training_curves,
            'convergence_comparison': convergence_comparison,
            'gradient_flow': gradient_flow,
            'activation_distributions': activation_distributions
        }

        return {
            'success': True,
            'metrics': metrics,
            'training_curves': training_curves,
            'convergence_comparison': convergence_comparison,
            'activation_distributions': activation_distributions,
            'gradient_flow': gradient_flow,
            'visualization_data': visualization_data,
            'execution_time_ms': execution_time_ms,
            'model_info': model_info,
            'parameters_used': {
                'momentum': self.request.momentum,
                'eps': self.request.eps,
                'affine': self.request.affine,
                'track_running_stats': self.request.track_running_stats,
                'epochs': self.request.epochs,
                'learning_rate': self.request.learning_rate,
                'batch_size': self.request.batch_size,
                'hidden_size': self.request.hidden_size
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information.

        Returns:
            Dictionary with model details
        """
        if self.model_with_bn is None:
            raise ValueError("Model must be trained first")

        # Count parameters
        params_with_bn = sum(p.numel() for p in self.model_with_bn.parameters())
        params_without_bn = sum(p.numel() for p in self.model_without_bn.parameters())

        return {
            'with_bn_parameters': int(params_with_bn),
            'without_bn_parameters': int(params_without_bn),
            'bn_overhead': int(params_with_bn - params_without_bn),
            'hidden_size': self.request.hidden_size,
            'num_layers': 3
        }
