"""Variational Autoencoder (VAE) model implementation using PyTorch.

This module provides a VAE implementation for learning latent representations
and generating new samples from learned distributions.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import time


class VAE(nn.Module):
    """Variational Autoencoder neural network.

    The VAE consists of an encoder that maps inputs to a latent distribution
    (mean and log-variance), and a decoder that reconstructs inputs from
    sampled latent vectors.

    Attributes:
        encoder: Encoder network (input -> hidden -> mu, log_var)
        decoder: Decoder network (latent -> hidden -> reconstruction)
        latent_dim: Dimension of latent space
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden: List[int],
        decoder_hidden: List[int]
    ):
        """Initialize VAE architecture.

        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
            encoder_hidden: List of hidden layer sizes for encoder
            decoder_hidden: List of hidden layer sizes for decoder
        """
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in encoder_hidden:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space layers (mean and log-variance)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in decoder_hidden:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Sigmoid for [0, 1] output

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor, shape (batch_size, input_dim)

        Returns:
            Tuple of (mu, log_var) tensors, each shape (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon.

        Args:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution

        Returns:
            Sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector, shape (batch_size, latent_dim)

        Returns:
            Reconstructed output, shape (batch_size, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss: reconstruction loss + beta * KL divergence.

    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log-variance of latent distribution
        beta: Weight for KL divergence term

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div


class VAEModel:
    """VAE model wrapper for training and inference.

    This class provides a high-level interface for training a VAE,
    generating samples, and visualizing the latent space.

    Attributes:
        model: VAE neural network
        optimizer: PyTorch optimizer
        device: Device to run model on (CPU or CUDA)
        training_history: Dictionary storing training metrics
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 20,
        encoder_hidden: List[int] = [128, 64],
        decoder_hidden: List[int] = [64, 128],
        learning_rate: float = 0.001,
        beta: float = 1.0,
        random_state: int = 42
    ):
        """Initialize VAE model.

        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
            encoder_hidden: Hidden layer sizes for encoder
            decoder_hidden: Hidden layer sizes for decoder
            learning_rate: Learning rate for optimizer
            beta: Weight for KL divergence term
            random_state: Random seed
        """
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = VAE(input_dim, latent_dim, encoder_hidden, decoder_hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.beta = beta
        self.training_history = {
            'total_loss': [],
            'recon_loss': [],
            'kl_div': []
        }

    def train(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 30,
        batch_size: int = 128
    ) -> Dict[str, Any]:
        """Train the VAE model.

        Args:
            X_train: Training data
            X_test: Test data
            y_train: Training labels (for visualization)
            y_test: Test labels (for visualization)
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Dictionary containing training results and metrics
        """
        start_time = time.time()

        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_div = 0.0

            for batch_data, _ in train_loader:
                batch_data = batch_data.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(batch_data)

                # Compute loss
                loss, recon_loss, kl_div = vae_loss(
                    recon_batch, batch_data, mu, logvar, self.beta
                )

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Accumulate losses
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_div += kl_div.item()

            # Store epoch metrics
            n_samples = len(X_train)
            self.training_history['total_loss'].append(epoch_loss / n_samples)
            self.training_history['recon_loss'].append(epoch_recon_loss / n_samples)
            self.training_history['kl_div'].append(epoch_kl_div / n_samples)

        training_time_ms = (time.time() - start_time) * 1000

        # Evaluate on test set
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            recon_test, mu_test, logvar_test = self.model(X_test_tensor)

            test_loss, test_recon_loss, test_kl_div = vae_loss(
                recon_test, X_test_tensor, mu_test, logvar_test, self.beta
            )

        return {
            'training_time_ms': training_time_ms,
            'final_train_loss': self.training_history['total_loss'][-1],
            'final_train_recon_loss': self.training_history['recon_loss'][-1],
            'final_train_kl_div': self.training_history['kl_div'][-1],
            'test_loss': test_loss.item() / len(X_test),
            'test_recon_loss': test_recon_loss.item() / len(X_test),
            'test_kl_div': test_kl_div.item() / len(X_test)
        }

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct inputs using the trained VAE.

        Args:
            X: Input data to reconstruct

        Returns:
            Reconstructed data
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            recon, _, _ = self.model(X_tensor)
            return recon.cpu().numpy()

    def encode_to_latent(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode inputs to latent space.

        Args:
            X: Input data

        Returns:
            Tuple of (mu, logvar) as numpy arrays
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            mu, logvar = self.model.encode(X_tensor)
            return mu.cpu().numpy(), logvar.cpu().numpy()

    def generate(self, n_samples: int = 10) -> np.ndarray:
        """Generate new samples from the latent space.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Generated samples as numpy array
        """
        self.model.eval()
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(n_samples, self.model.latent_dim).to(self.device)
            generated = self.model.decode(z)
            return generated.cpu().numpy()

    def get_latent_2d_projection(
        self,
        X: np.ndarray,
        method: str = 'pca'
    ) -> np.ndarray:
        """Get 2D projection of latent representations.

        Args:
            X: Input data
            method: Projection method ('pca' or 'first2')

        Returns:
            2D latent representations
        """
        mu, _ = self.encode_to_latent(X)

        if method == 'pca' and mu.shape[1] > 2:
            pca = PCA(n_components=2)
            return pca.fit_transform(mu)
        else:
            # Just use first 2 dimensions
            return mu[:, :2]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture and parameter information.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'latent_dim': self.model.latent_dim,
            'input_dim': self.model.input_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'beta': self.beta
        }
