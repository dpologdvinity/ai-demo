"""Feedforward Neural Network (MLP) implementation using scikit-learn."""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss
)


class MLPModel:
    """MLP model wrapper for training and evaluation.

    This class wraps scikit-learn's MLPClassifier and provides a consistent
    interface for training, evaluation, and visualization data extraction.

    Attributes:
        model: The MLPClassifier instance
        training_history: Records of loss and accuracy per iteration
        hidden_layers: Architecture of hidden layers
        learning_rate: Initial learning rate
        activation: Activation function used
        batch_size: Minibatch size for training
    """

    def __init__(
        self,
        hidden_layers: List[int] = [64, 32],
        learning_rate: float = 0.001,
        activation: str = 'relu',
        batch_size: int = 32,
        random_state: int = 42
    ):
        """Initialize MLP model with configuration.

        Args:
            hidden_layers: Sizes of hidden layers
            learning_rate: Initial learning rate
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'logistic')
            batch_size: Size of minibatches
            random_state: Random seed for reproducibility
        """
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.activation = activation if activation != 'sigmoid' else 'logistic'
        self.batch_size = batch_size
        self.random_state = random_state

        # Initialize the model
        self.model: Optional[MLPClassifier] = None
        self.training_history: Dict[str, List[float]] = {
            'iterations': [],
            'loss': [],
            'accuracy': []
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        max_iter: int = 100
    ) -> Dict[str, Any]:
        """Train the MLP model.

        Args:
            X_train: Training features (N, D)
            y_train: Training labels (N,)
            X_test: Test features (M, D)
            y_test: Test labels (M,)
            max_iter: Maximum number of training iterations

        Returns:
            Dictionary with training information
        """
        start_time = time.time()

        # Initialize model with warm_start to track training progress
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            learning_rate_init=self.learning_rate,
            batch_size=self.batch_size,
            max_iter=1,  # Train one iteration at a time
            warm_start=True,  # Keep weights between .fit() calls
            random_state=self.random_state,
            early_stopping=False,
            verbose=False
        )

        # Track training progress by training one iteration at a time
        for iteration in range(max_iter):
            # Train for one iteration
            self.model.fit(X_train, y_train)

            # Calculate training loss and accuracy
            try:
                # Get predicted probabilities for loss calculation
                y_pred_proba = self.model.predict_proba(X_train)
                loss = log_loss(y_train, y_pred_proba)
            except Exception:
                # Fallback to using the model's loss_ attribute
                loss = self.model.loss_ if hasattr(self.model, 'loss_') else 0.0

            # Calculate training accuracy
            y_pred = self.model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)

            # Store metrics
            self.training_history['iterations'].append(iteration + 1)
            self.training_history['loss'].append(float(loss))
            self.training_history['accuracy'].append(float(accuracy))

            # Check for early convergence
            if iteration > 10 and abs(loss - self.training_history['loss'][-2]) < 1e-6:
                break

        training_time = time.time() - start_time

        # Final evaluation
        final_train_accuracy = float(accuracy_score(y_train, self.model.predict(X_train)))
        final_test_accuracy = float(accuracy_score(y_test, self.model.predict(X_test)))

        return {
            'training_time_ms': training_time * 1000,
            'final_train_accuracy': final_train_accuracy,
            'final_test_accuracy': final_test_accuracy,
            'iterations_trained': len(self.training_history['iterations'])
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict_proba(X)

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

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information.

        Returns:
            Dictionary with model details
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        # Calculate total parameters
        total_params = 0
        layer_sizes = [self.model.n_features_in_] + list(self.hidden_layers) + [self.model.n_outputs_]

        for i in range(len(layer_sizes) - 1):
            # Weights: layer_sizes[i] * layer_sizes[i+1]
            # Biases: layer_sizes[i+1]
            total_params += layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]

        # Get layer info
        layers_info = []
        layers_info.append({
            'type': 'Input',
            'units': int(self.model.n_features_in_)
        })

        for i, size in enumerate(self.hidden_layers):
            layers_info.append({
                'type': f'Hidden_{i+1}',
                'units': int(size),
                'activation': self.activation
            })

        layers_info.append({
            'type': 'Output',
            'units': int(self.model.n_outputs_),
            'activation': 'softmax'
        })

        return {
            'hidden_layers': list(self.hidden_layers),
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'total_parameters': total_params,
            'input_size': int(self.model.n_features_in_),
            'output_size': int(self.model.n_outputs_),
            'layers': layers_info
        }

    def get_network_architecture(self) -> Dict[str, Any]:
        """Get network architecture for visualization.

        Returns:
            Dictionary with network structure
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        model_info = self.get_model_info()

        return {
            'layers': model_info['layers'],
            'total_parameters': model_info['total_parameters'],
            'connections': [
                {
                    'from_layer': i,
                    'to_layer': i + 1,
                    'weights': model_info['layers'][i]['units'] * model_info['layers'][i + 1]['units']
                }
                for i in range(len(model_info['layers']) - 1)
            ]
        }
