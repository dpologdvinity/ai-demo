"""Learning Rate Scheduling model implementation."""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple

from .schema import LearningRateSchedulingRequest, LearningRateSchedulingResponse


class SimpleClassifier(nn.Module):
    """Simple neural network for classification."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        """Initialize classifier.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output logits
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LearningRateSchedulingModel:
    """Model for demonstrating learning rate scheduling."""

    def __init__(self, request: LearningRateSchedulingRequest):
        """Initialize model with request parameters.

        Args:
            request: Request containing scheduling parameters
        """
        self.request = request
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set random seeds
        np.random.seed(request.random_state)
        torch.manual_seed(request.random_state)

        # Generate synthetic data
        self.X_train, self.X_test, self.y_train, self.y_test = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic classification data.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=self.request.random_state,
            class_sep=1.0
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.request.random_state
        )

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)

        return X_train, X_test, y_train, y_test

    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        schedule_type: str
    ) -> Any:
        """Create learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer
            schedule_type: Type of schedule to create

        Returns:
            Learning rate scheduler
        """
        if schedule_type == 'step':
            return StepLR(
                optimizer,
                step_size=self.request.step_size,
                gamma=self.request.gamma
            )
        elif schedule_type == 'exponential':
            return ExponentialLR(optimizer, gamma=self.request.gamma)
        elif schedule_type == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=self.request.epochs,
                eta_min=self.request.min_lr
            )
        elif schedule_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.request.gamma,
                patience=self.request.patience
            )
        elif schedule_type == 'cyclic':
            return CyclicLR(
                optimizer,
                base_lr=self.request.initial_lr,
                max_lr=self.request.max_lr,
                step_size_up=self.request.step_size,
                mode='triangular'
            )
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def _train_with_schedule(
        self,
        schedule_type: str
    ) -> Tuple[List[float], List[float], float]:
        """Train model with a specific learning rate schedule.

        Args:
            schedule_type: Type of learning rate schedule

        Returns:
            Tuple of (lr_history, loss_history, training_time)
        """
        start_time = time.time()

        # Initialize model and optimizer
        model = SimpleClassifier().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.request.initial_lr)
        criterion = nn.CrossEntropyLoss()

        # Create scheduler
        scheduler = self._create_scheduler(optimizer, schedule_type)

        lr_history = []
        loss_history = []

        # Training loop
        for epoch in range(self.request.epochs):
            model.train()

            # Forward pass
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record learning rate and loss
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            loss_history.append(loss.item())

            # Update learning rate
            if schedule_type == 'reduce_on_plateau':
                scheduler.step(loss)
            elif schedule_type == 'cyclic':
                scheduler.step()
            else:
                scheduler.step()

        training_time = (time.time() - start_time) * 1000  # Convert to ms

        return lr_history, loss_history, training_time

    def _compute_convergence_metrics(
        self,
        loss_history: List[float],
        training_time: float,
        convergence_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Compute convergence metrics for a training run.

        Args:
            loss_history: Loss values over epochs
            training_time: Total training time in milliseconds
            convergence_threshold: Loss threshold to consider converged

        Returns:
            Dictionary of convergence metrics
        """
        final_loss = loss_history[-1]

        # Find convergence epoch (when loss first drops below threshold)
        convergence_epoch = None
        for epoch, loss in enumerate(loss_history):
            if loss < convergence_threshold:
                convergence_epoch = epoch + 1
                break

        # If never converged, use total epochs
        if convergence_epoch is None:
            convergence_epoch = len(loss_history)

        # Compute stability (variance in last 20% of training)
        last_20_percent = int(len(loss_history) * 0.2)
        stability = float(np.var(loss_history[-last_20_percent:]))

        return {
            'final_loss': float(final_loss),
            'convergence_epoch': convergence_epoch,
            'training_time': float(training_time),
            'stability': stability,
            'min_loss': float(min(loss_history)),
            'avg_loss_last_10': float(np.mean(loss_history[-10:]))
        }

    def train(self) -> LearningRateSchedulingResponse:
        """Train models with all learning rate schedules.

        Returns:
            Response containing training results and visualizations
        """
        try:
            start_time = time.time()

            # Train with all schedule types
            schedule_types = ['step', 'exponential', 'cosine', 'reduce_on_plateau', 'cyclic']

            schedule_data = {}
            loss_data = {}
            convergence_metrics = {}
            comparison_table = []

            for schedule_type in schedule_types:
                print(f"Training with {schedule_type} schedule...")

                lr_history, loss_history, training_time = self._train_with_schedule(
                    schedule_type
                )

                schedule_data[schedule_type] = lr_history
                loss_data[schedule_type] = loss_history

                metrics = self._compute_convergence_metrics(loss_history, training_time)
                convergence_metrics[schedule_type] = metrics

                # Add to comparison table
                comparison_table.append({
                    'schedule': schedule_type.replace('_', ' ').title(),
                    'final_loss': metrics['final_loss'],
                    'convergence_epoch': metrics['convergence_epoch'],
                    'training_time': metrics['training_time'],
                    'stability': metrics['stability'],
                    'min_loss': metrics['min_loss']
                })

            # Sort comparison table by final loss (best first)
            comparison_table.sort(key=lambda x: x['final_loss'])

            # Prepare visualization data
            visualization_data = {
                'epochs': list(range(1, self.request.epochs + 1)),
                'schedule_types': schedule_types,
                'convergence_threshold': 0.1,
                'best_schedule': comparison_table[0]['schedule']
            }

            execution_time_ms = (time.time() - start_time) * 1000

            return LearningRateSchedulingResponse(
                success=True,
                schedule_data=schedule_data,
                loss_data=loss_data,
                convergence_metrics=convergence_metrics,
                comparison_table=comparison_table,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                parameters_used={
                    'schedule_type': self.request.schedule_type,
                    'initial_lr': self.request.initial_lr,
                    'step_size': self.request.step_size,
                    'gamma': self.request.gamma,
                    'epochs': self.request.epochs,
                    'min_lr': self.request.min_lr,
                    'max_lr': self.request.max_lr,
                    'patience': self.request.patience
                }
            )

        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")


def get_model_info() -> Dict[str, Any]:
    """Get information about the learning rate scheduling model.

    Returns:
        Dictionary containing model architecture and schedule information
    """
    return {
        'architecture': 'Simple Feedforward Neural Network',
        'layers': [
            {'type': 'Input', 'size': 20},
            {'type': 'Linear', 'size': 64},
            {'type': 'ReLU'},
            {'type': 'Linear', 'size': 64},
            {'type': 'ReLU'},
            {'type': 'Linear', 'size': 2},
        ],
        'optimizer': 'SGD',
        'loss_function': 'CrossEntropyLoss',
        'schedules': {
            'step': 'Decays LR by gamma every step_size epochs',
            'exponential': 'Decays LR exponentially by gamma each epoch',
            'cosine': 'Cosine annealing from initial_lr to min_lr',
            'reduce_on_plateau': 'Reduces LR when loss plateaus',
            'cyclic': 'Cycles LR between base_lr and max_lr'
        }
    }
