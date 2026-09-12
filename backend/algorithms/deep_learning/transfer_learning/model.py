"""Transfer Learning model implementation using PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time
import logging
from sklearn.metrics import confusion_matrix, accuracy_score

from .schema import (
    TransferLearningRequest,
    TransferLearningResponse,
    LayerInfo,
    StrategyComparison
)
from .data import create_synthetic_image_dataset, normalize_images

logger = logging.getLogger(__name__)


class TransferLearningModel:
    """Transfer Learning model using pre-trained networks."""

    def __init__(
        self,
        base_model: str = "resnet18",
        num_classes: int = 5,
        strategy: str = "fine_tune",
        freeze_layers: str = "auto",
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """Initialize transfer learning model.

        Args:
            base_model: Pre-trained model to use
            num_classes: Number of output classes
            strategy: Transfer learning strategy
            freeze_layers: Which layers to freeze
            learning_rate: Learning rate for training
            random_state: Random seed
        """
        self.base_model_name = base_model
        self.num_classes = num_classes
        self.strategy = strategy
        self.freeze_layers_config = freeze_layers
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained model
        self.model = self._load_pretrained_model()

        # Apply freezing strategy
        self._apply_freezing_strategy()

        # Training history
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def _load_pretrained_model(self) -> nn.Module:
        """Load pre-trained model and modify final layer."""
        logger.info(f"Loading pre-trained {self.base_model_name}")

        # Load model based on architecture
        if self.base_model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)

        elif self.base_model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)

        elif self.base_model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)

        elif self.base_model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)

        else:
            raise ValueError(f"Unknown model: {self.base_model_name}")

        return model.to(self.device)

    def _apply_freezing_strategy(self):
        """Apply layer freezing based on strategy."""
        if self.strategy == 'feature_extraction':
            # Freeze all layers except final classifier
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze final layer
            if self.base_model_name.startswith('resnet'):
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif self.base_model_name == 'mobilenet_v2':
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            elif self.base_model_name == 'efficientnet_b0':
                for param in self.model.classifier.parameters():
                    param.requires_grad = True

            logger.info("Strategy: Feature extraction (all layers frozen except classifier)")

        elif self.strategy == 'fine_tune':
            # Determine freeze configuration
            if self.freeze_layers_config == 'auto':
                # Auto: freeze early layers, train later layers
                self._freeze_early_layers()
            elif self.freeze_layers_config == 'early':
                self._freeze_early_layers()
            elif self.freeze_layers_config == 'most':
                self._freeze_most_layers()
            elif self.freeze_layers_config == 'all_but_last':
                self._freeze_all_but_last()
            elif self.freeze_layers_config == 'none':
                # Don't freeze anything
                pass

            logger.info(f"Strategy: Fine-tuning with freeze_layers={self.freeze_layers_config}")

        elif self.strategy == 'full_train':
            # Train all layers (no freezing)
            logger.info("Strategy: Full training (no layers frozen)")

    def _freeze_early_layers(self):
        """Freeze early convolutional layers."""
        if self.base_model_name.startswith('resnet'):
            # Freeze conv1, bn1, layer1, layer2
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2']):
                    param.requires_grad = False

        elif self.base_model_name == 'mobilenet_v2':
            # Freeze first half of features
            for i, child in enumerate(self.model.features.children()):
                if i < len(list(self.model.features.children())) // 2:
                    for param in child.parameters():
                        param.requires_grad = False

        elif self.base_model_name == 'efficientnet_b0':
            # Freeze first half of features
            for i, child in enumerate(self.model.features.children()):
                if i < len(list(self.model.features.children())) // 2:
                    for param in child.parameters():
                        param.requires_grad = False

    def _freeze_most_layers(self):
        """Freeze most layers, train only last few."""
        if self.base_model_name.startswith('resnet'):
            # Freeze everything except layer4 and fc
            for name, param in self.model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

        elif self.base_model_name in ['mobilenet_v2', 'efficientnet_b0']:
            # Freeze all features except last block
            for i, child in enumerate(self.model.features.children()):
                if i < len(list(self.model.features.children())) - 2:
                    for param in child.parameters():
                        param.requires_grad = False

    def _freeze_all_but_last(self):
        """Freeze all layers except the final classifier."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier
        if self.base_model_name.startswith('resnet'):
            for param in self.model.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """Train the transfer learning model.

        Args:
            X_train: Training images (N, H, W, C)
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training results dictionary
        """
        logger.info(f"Training transfer learning model for {epochs} epochs")

        # Normalize images
        X_train_norm = normalize_images(X_train)
        X_val_norm = normalize_images(X_val)

        # Convert to PyTorch tensors (N, C, H, W)
        X_train_tensor = torch.FloatTensor(X_train_norm).permute(0, 3, 1, 2)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_norm).permute(0, 3, 1, 2)
        y_val_tensor = torch.LongTensor(y_val)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total

            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)

            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        return {
            'final_train_loss': train_loss,
            'final_train_accuracy': train_accuracy,
            'final_val_loss': val_loss,
            'final_val_accuracy': val_accuracy
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input images.

        Args:
            X: Input images (N, H, W, C)

        Returns:
            Predicted class labels
        """
        self.model.eval()

        # Normalize and convert to tensor
        X_norm = normalize_images(X)
        X_tensor = torch.FloatTensor(X_norm).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Input images (N, H, W, C)

        Returns:
            Prediction probabilities (N, num_classes)
        """
        self.model.eval()

        # Normalize and convert to tensor
        X_norm = normalize_images(X)
        X_tensor = torch.FloatTensor(X_norm).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            X: Test images
            y: True labels

        Returns:
            Evaluation metrics
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)

        logger.info(f"Test accuracy: {accuracy:.4f}")

        return {
            'accuracy': accuracy,
            'num_samples': len(y)
        }

    def get_layer_info(self) -> List[LayerInfo]:
        """Get information about model layers.

        Returns:
            List of layer information
        """
        layer_info = []

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                # Leaf module with parameters
                num_params = sum(p.numel() for p in module.parameters())
                trainable = any(p.requires_grad for p in module.parameters())
                frozen = not trainable

                layer_info.append(LayerInfo(
                    name=name,
                    trainable=trainable,
                    num_params=num_params,
                    frozen=frozen
                ))

        return layer_info

    def count_parameters(self) -> Tuple[int, int, int]:
        """Count model parameters.

        Returns:
            Tuple of (total_params, trainable_params, frozen_params)
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return total_params, trainable_params, frozen_params

    def extract_features(self, X: np.ndarray, layer_name: str = 'avgpool') -> np.ndarray:
        """Extract features from a specific layer.

        Args:
            X: Input images
            layer_name: Name of layer to extract from

        Returns:
            Extracted features
        """
        self.model.eval()

        # Normalize and convert to tensor
        X_norm = normalize_images(X)
        X_tensor = torch.FloatTensor(X_norm).permute(0, 3, 1, 2).to(self.device)

        features = []

        def hook_fn(module, input, output):
            features.append(output.detach().cpu().numpy())

        # Register hook
        target_layer = None
        for name, module in self.model.named_modules():
            if layer_name in name:
                target_layer = module
                break

        if target_layer is None:
            logger.warning(f"Layer {layer_name} not found, using final features")
            # Use features before classifier
            if self.base_model_name.startswith('resnet'):
                target_layer = self.model.avgpool
            else:
                target_layer = list(self.model.features.children())[-1]

        handle = target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.model(X_tensor)

        handle.remove()

        return features[0] if features else None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model details
        """
        total_params, trainable_params, frozen_params = self.count_parameters()

        return {
            'base_model': self.base_model_name,
            'num_classes': self.num_classes,
            'strategy': self.strategy,
            'freeze_layers': self.freeze_layers_config,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'learning_rate': self.learning_rate
        }


def run_transfer_learning(request: TransferLearningRequest) -> TransferLearningResponse:
    """Run transfer learning training.

    Args:
        request: Transfer learning request parameters

    Returns:
        Transfer learning response with results
    """
    start_time = time.time()

    # Load dataset
    data = create_synthetic_image_dataset(
        dataset_name=request.dataset,
        num_classes=5,
        samples_per_class=30,
        img_size=224,
        random_state=request.random_state
    )

    # Initialize model
    model = TransferLearningModel(
        base_model=request.base_model,
        num_classes=data['num_classes'],
        strategy=request.strategy,
        freeze_layers=request.freeze_layers,
        learning_rate=request.learning_rate,
        random_state=request.random_state
    )

    # Train model
    train_results = model.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        epochs=request.epochs,
        batch_size=request.batch_size
    )

    # Evaluate on test set
    test_results = model.evaluate(data['X_test'], data['y_test'])

    # Get predictions
    y_pred = model.predict(data['X_test'])
    y_proba = model.predict_proba(data['X_test'])

    # Confusion matrix
    cm = confusion_matrix(data['y_test'], y_pred)

    # Get layer information
    layer_info = model.get_layer_info()

    # Sample predictions (first 10)
    sample_predictions = []
    for i in range(min(10, len(data['X_test']))):
        sample_predictions.append({
            'image_index': i,
            'true_label': data['class_names'][data['y_test'][i]],
            'predicted_label': data['class_names'][y_pred[i]],
            'confidence': float(y_proba[i, y_pred[i]]),
            'top_3_probs': y_proba[i].tolist()
        })

    # Extract features for visualization
    features = model.extract_features(data['X_test'][:5])

    # Prepare visualization data
    visualization_data = {
        'training_curves': {
            'epochs': list(range(1, request.epochs + 1)),
            'train_loss': model.training_history['train_loss'],
            'train_accuracy': model.training_history['train_accuracy'],
            'val_loss': model.training_history['val_loss'],
            'val_accuracy': model.training_history['val_accuracy']
        },
        'confusion_matrix': cm.tolist(),
        'class_names': data['class_names'],
        'num_samples': {
            'train': data['n_train'],
            'val': data['n_val'],
            'test': data['n_test']
        },
        'convergence_comparison': {
            'strategy': request.strategy,
            'convergence_epoch': _find_convergence_epoch(model.training_history['val_accuracy'])
        }
    }

    # Get model info
    model_info = model.get_model_info()

    execution_time_ms = (time.time() - start_time) * 1000

    # Prepare metrics
    metrics = {
        'train_accuracy': train_results['final_train_accuracy'],
        'train_loss': train_results['final_train_loss'],
        'val_accuracy': train_results['final_val_accuracy'],
        'val_loss': train_results['final_val_loss'],
        'test_accuracy': test_results['accuracy']
    }

    return TransferLearningResponse(
        success=True,
        metrics=metrics,
        training_history={
            'epochs': list(range(1, request.epochs + 1)),
            'train_loss': model.training_history['train_loss'],
            'train_accuracy': model.training_history['train_accuracy'],
            'val_loss': model.training_history['val_loss'],
            'val_accuracy': model.training_history['val_accuracy']
        },
        confusion_matrix=cm.tolist(),
        layer_info=layer_info,
        sample_predictions=sample_predictions,
        visualization_data=visualization_data,
        execution_time_ms=execution_time_ms,
        model_info=model_info,
        parameters_used={
            'base_model': request.base_model,
            'strategy': request.strategy,
            'freeze_layers': request.freeze_layers,
            'learning_rate': request.learning_rate,
            'epochs': request.epochs,
            'batch_size': request.batch_size,
            'dataset': request.dataset
        }
    )


def _find_convergence_epoch(val_accuracy: List[float], threshold: float = 0.01) -> int:
    """Find epoch where model converged (accuracy stops improving significantly).

    Args:
        val_accuracy: Validation accuracy history
        threshold: Improvement threshold

    Returns:
        Convergence epoch
    """
    for i in range(1, len(val_accuracy)):
        if abs(val_accuracy[i] - val_accuracy[i-1]) < threshold:
            return i + 1

    return len(val_accuracy)
