"""Autoencoder implementation using PyTorch.

This module provides an autoencoder implementation for unsupervised feature
learning and dimensionality reduction on MNIST digits. The autoencoder consists
of an encoder that compresses images to a latent space and a decoder that
reconstructs the original images.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time


class Encoder(nn.Module):
    """Encoder network that compresses images to latent space.

    The encoder takes an image as input and compresses it through hidden layers
    to a low-dimensional latent representation.

    Architecture:
        Input (784) -> Hidden (hidden_dim) with ReLU -> Latent (latent_dim)

    Attributes:
        model: Sequential neural network model
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, latent_dim: int = 32):
        """Initialize Encoder network.

        Args:
            input_dim: Input dimension (64 for 8x8 images). Default: 64
            hidden_dim: Size of the hidden layer. Default: 128
            latent_dim: Dimension of the latent space. Default: 32
        """
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Input images, shape (batch_size, 64)

        Returns:
            Latent representations, shape (batch_size, latent_dim)
        """
        return self.model(x)


class Decoder(nn.Module):
    """Decoder network that reconstructs images from latent space.

    The decoder takes a latent representation and expands it through hidden layers
    to reconstruct the original image.

    Architecture:
        Latent (latent_dim) -> Hidden (hidden_dim) with ReLU -> Output (784) with Sigmoid

    Attributes:
        model: Sequential neural network model
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128, output_dim: int = 64):
        """Initialize Decoder network.

        Args:
            latent_dim: Dimension of the latent space. Default: 32
            hidden_dim: Size of the hidden layer. Default: 128
            output_dim: Output dimension (64 for 8x8 images). Default: 64
        """
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            z: Latent representations, shape (batch_size, latent_dim)

        Returns:
            Reconstructed images, shape (batch_size, 64)
        """
        return self.model(z)


class AutoencoderModel:
    """Autoencoder for unsupervised feature learning and dimensionality reduction.

    This class implements an autoencoder that learns to compress MNIST digit images
    into a low-dimensional latent space and reconstruct them. The model uses MSE
    (Mean Squared Error) as the reconstruction loss.

    Attributes:
        encoder: Encoder network
        decoder: Decoder network
        latent_dim: Dimension of the latent space
        hidden_dim: Size of hidden layers
        device: Device to run computations on (CPU or CUDA)
        training_history: Dictionary storing loss history during training
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """Initialize Autoencoder model.

        Args:
            latent_dim: Dimension of the latent space. Default: 32
            hidden_dim: Size of hidden layers in encoder/decoder. Default: 128
            learning_rate: Learning rate for Adam optimizer. Default: 0.001
            random_state: Random seed for reproducibility. Default: 42

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if latent_dim < 2 or latent_dim > 128:
            raise ValueError(f"latent_dim must be between 2 and 128, got {latent_dim}")
        if hidden_dim < 64 or hidden_dim > 512:
            raise ValueError(f"hidden_dim must be between 64 and 512, got {hidden_dim}")

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(64, hidden_dim, latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim, hidden_dim, 64).to(self.device)

        # Optimizer (combined for both encoder and decoder)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )

        # Loss function (MSE for reconstruction)
        self.criterion = nn.MSELoss()

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }

    def train(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """Train the autoencoder on digit images.

        Trains the encoder and decoder to minimize reconstruction loss (MSE)
        between original and reconstructed images.

        Args:
            X_train: Training data, shape (n_train, 64)
            X_test: Test data for validation, shape (n_test, 64)
            epochs: Number of training epochs. Default: 50
            batch_size: Batch size for training. Default: 64

        Returns:
            Dictionary containing:
                - training_history: Loss history per epoch
                - final_train_loss: Final training loss
                - final_val_loss: Final validation loss

        Raises:
            ValueError: If X_train or X_test is empty or has wrong shape.
        """
        if X_train.size == 0 or X_test.size == 0:
            raise ValueError("Training or test data cannot be empty")
        if X_train.ndim != 2 or X_train.shape[1] != 64:
            raise ValueError(f"X_train must have shape (n_samples, 64), got {X_train.shape}")
        if X_test.ndim != 2 or X_test.shape[1] != 64:
            raise ValueError(f"X_test must have shape (n_samples, 64), got {X_test.shape}")

        # Prepare data loaders
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)

        train_dataset = TensorDataset(X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.encoder.train()
            self.decoder.train()
            train_loss = 0.0
            n_batches = 0

            for batch_x, in train_loader:
                # Forward pass
                latent = self.encoder(batch_x)
                reconstructed = self.decoder(latent)

                # Compute loss
                loss = self.criterion(reconstructed, batch_x)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / n_batches

            # Validation phase
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                latent = self.encoder(X_test_tensor)
                reconstructed = self.decoder(latent)
                val_loss = self.criterion(reconstructed, X_test_tensor).item()

            # Record history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)

        return {
            'training_history': self.training_history,
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1]
        }

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode images to latent space.

        Args:
            X: Images to encode, shape (n_samples, 64)

        Returns:
            Latent representations, shape (n_samples, latent_dim)
        """
        self.encoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            latent = self.encoder(X_tensor)
            return latent.cpu().numpy()

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent representations to images.

        Args:
            z: Latent representations, shape (n_samples, latent_dim)

        Returns:
            Reconstructed images, shape (n_samples, 64)
        """
        self.decoder.eval()
        with torch.no_grad():
            z_tensor = torch.FloatTensor(z).to(self.device)
            reconstructed = self.decoder(z_tensor)
            return reconstructed.cpu().numpy()

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct images through encode-decode pipeline.

        Args:
            X: Original images, shape (n_samples, 64)

        Returns:
            Reconstructed images, shape (n_samples, 64)
        """
        latent = self.encode(X)
        return self.decode(latent)

    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluate reconstruction quality on test data.

        Args:
            X: Test images, shape (n_samples, 64)

        Returns:
            Dictionary containing:
                - reconstruction_loss: MSE reconstruction loss
                - avg_pixel_error: Average per-pixel error
        """
        reconstructed = self.reconstruct(X)
        mse = np.mean((X - reconstructed) ** 2)
        mae = np.mean(np.abs(X - reconstructed))

        return {
            'reconstruction_loss': float(mse),
            'avg_pixel_error': float(mae)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.

        Returns:
            Dictionary containing model architecture and parameter counts.
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        return {
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'encoder_params': int(encoder_params),
            'decoder_params': int(decoder_params),
            'total_params': int(encoder_params + decoder_params),
            'device': str(self.device)
        }

    def count_parameters(self) -> Tuple[int, int]:
        """Count trainable parameters in both networks.

        Returns:
            Tuple of (encoder_params, decoder_params)
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return int(encoder_params), int(decoder_params)
