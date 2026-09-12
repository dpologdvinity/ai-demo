"""Convolutional Neural Network (CNN) implementation using PyTorch."""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class CNNClassifier(nn.Module):
    """PyTorch CNN architecture with configurable convolutional layers.

    This network consists of:
    - Multiple Conv2d + BatchNorm2d + ReLU + MaxPool2d layers
    - Adaptive pooling to handle different input sizes
    - Fully connected layers with dropout
    - Output layer for classification

    Attributes:
        conv_layers: Sequential container of convolutional layers
        fc_layers: Sequential container of fully connected layers
        feature_maps: Stored feature maps from first conv layer for visualization
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        conv_filters: List[int] = [16, 32],
        kernel_size: int = 3,
        dropout: float = 0.5
    ):
        """Initialize CNN architecture.

        Args:
            in_channels: Number of input channels (1 for grayscale)
            num_classes: Number of output classes
            conv_filters: List of filter counts for each conv layer
            kernel_size: Size of convolutional kernel
            dropout: Dropout rate for regularization
        """
        super(CNNClassifier, self).__init__()

        self.feature_maps = None

        # Build convolutional layers
        layers = []
        prev_filters = in_channels

        for i, num_filters in enumerate(conv_filters):
            layers.extend([
                nn.Conv2d(prev_filters, num_filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            prev_filters = num_filters

        self.conv_layers = nn.Sequential(*layers)

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_filters, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, extract_features: bool = False) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)
            extract_features: If True, store feature maps from first conv layer

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # First conv layer with feature extraction
        x = self.conv_layers[0](x)  # Conv2d
        if extract_features:
            self.feature_maps = x.detach().cpu()

        # Rest of conv layers
        for layer in self.conv_layers[1:]:
            x = layer(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Fully connected layers
        x = self.fc_layers(x)

        return x


class CNNModel:
    """CNN model wrapper for training and evaluation.

    Handles the complete training pipeline including data loading,
    training loop, evaluation, and visualization data extraction.

    Attributes:
        model: The CNNClassifier network
        device: Compute device (cuda or cpu)
        optimizer: Optimization algorithm
        criterion: Loss function
        training_history: Records of loss and accuracy per epoch
    """

    def __init__(
        self,
        conv_filters: List[int] = [16, 32],
        kernel_size: int = 3,
        learning_rate: float = 0.001,
        dropout: float = 0.5
    ):
        """Initialize CNN model with configuration.

        Args:
            conv_filters: Filters per convolutional layer
            kernel_size: Convolution kernel size
            learning_rate: Learning rate for optimizer
            dropout: Dropout rate for regularization
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[CNNClassifier] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.CrossEntropyLoss()
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Store hyperparameters
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.dropout = dropout

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 15,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Train the CNN model.

        Args:
            X_train: Training features (N, H, W)
            y_train: Training labels (N,)
            X_test: Test features
            y_test: Test labels
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Dictionary with training information
        """
        start_time = time.time()

        # Reshape for CNN: (N, C, H, W)
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(-1, 1, X_train.shape[1], X_train.shape[2])
            X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        num_classes = len(np.unique(y_train))
        self.model = CNNClassifier(
            in_channels=1,
            num_classes=num_classes,
            conv_filters=self.conv_filters,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate_epoch(test_loader)

            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)

        training_time = time.time() - start_time

        return {
            'training_time_ms': training_time * 1000,
            'final_train_accuracy': self.training_history['train_accuracy'][-1],
            'final_val_accuracy': self.training_history['val_accuracy'][-1],
            'epochs_trained': epochs
        }

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch.

        Args:
            test_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(test_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()

        # Reshape if needed
        if len(X.shape) == 3:
            X = X.reshape(-1, 1, X.shape[1], X.shape[2])

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)

        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

    def extract_feature_maps(self, X: np.ndarray, num_samples: int = 5) -> np.ndarray:
        """Extract feature maps from first convolutional layer.

        Args:
            X: Input samples
            num_samples: Number of samples to extract features from

        Returns:
            Feature maps array of shape (num_samples, num_filters, H, W)
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        self.model.eval()

        # Take only num_samples
        X_sample = X[:num_samples]

        # Reshape if needed
        if len(X_sample.shape) == 3:
            X_sample = X_sample.reshape(-1, 1, X_sample.shape[1], X_sample.shape[2])

        X_tensor = torch.FloatTensor(X_sample).to(self.device)

        with torch.no_grad():
            # Forward pass with feature extraction
            _ = self.model(X_tensor, extract_features=True)

        return self.model.feature_maps.numpy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information.

        Returns:
            Dictionary with model details
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'conv_filters': self.conv_filters,
            'kernel_size': self.kernel_size,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }
