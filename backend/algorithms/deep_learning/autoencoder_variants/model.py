"""Autoencoder Variants implementation using PyTorch.

This module implements four autoencoder variants:
1. Vanilla: Standard encoder-decoder
2. Denoising: Trained to remove noise
3. Sparse: L1 regularization on activations
4. Contractive: Penalty on Jacobian
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    """Encoder network: 784 → 256 → 128 → latent_dim"""

    def __init__(self, input_dim: int = 64, latent_dim: int = 32):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        z = self.fc3(h2)
        return z


class Decoder(nn.Module):
    """Decoder network: latent_dim → 128 → 256 → 784"""

    def __init__(self, latent_dim: int = 32, output_dim: int = 64):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h1 = self.relu(self.fc1(z))
        h2 = self.relu(self.fc2(h1))
        x_recon = self.sigmoid(self.fc3(h2))
        return x_recon


class VanillaAutoencoder(nn.Module):
    """Standard autoencoder with encoder-decoder architecture."""

    def __init__(self, input_dim: int = 64, latent_dim: int = 32):
        super(VanillaAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class DenoisingAutoencoder(nn.Module):
    """Autoencoder trained to remove noise from corrupted inputs."""

    def __init__(self, input_dim: int = 64, latent_dim: int = 32):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class SparseAutoencoder(nn.Module):
    """Autoencoder with L1 regularization on latent activations."""

    def __init__(self, input_dim: int = 64, latent_dim: int = 32):
        super(SparseAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class ContractiveAutoencoder(nn.Module):
    """Autoencoder with penalty on Jacobian (contractive penalty)."""

    def __init__(self, input_dim: int = 64, latent_dim: int = 32):
        super(ContractiveAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class AutoencoderVariantsModel:
    """Main model class for training and evaluating autoencoder variants."""

    def __init__(
        self,
        variant: str = 'vanilla',
        latent_dim: int = 32,
        learning_rate: float = 0.001,
        noise_factor: float = 0.3,
        sparsity_weight: float = 0.001,
        random_state: int = 42
    ):
        """Initialize autoencoder variant.

        Args:
            variant: Type of autoencoder ('vanilla', 'denoising', 'sparse', 'contractive')
            latent_dim: Dimension of latent space
            learning_rate: Learning rate for optimizer
            noise_factor: Noise level for denoising (0.0-0.5)
            sparsity_weight: Weight for sparsity penalty
            random_state: Random seed
        """
        # Validate parameters
        if variant not in ['vanilla', 'denoising', 'sparse', 'contractive']:
            raise ValueError(f"Invalid variant: {variant}. Must be one of: vanilla, denoising, sparse, contractive")
        if latent_dim < 2 or latent_dim > 128:
            raise ValueError(f"latent_dim must be between 2 and 128, got {latent_dim}")
        if noise_factor < 0 or noise_factor > 0.5:
            raise ValueError(f"noise_factor must be between 0 and 0.5, got {noise_factor}")

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Configuration
        self.variant = variant
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.noise_factor = noise_factor
        self.sparsity_weight = sparsity_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model based on variant
        if variant == 'vanilla':
            self.model = VanillaAutoencoder(64, latent_dim).to(self.device)
        elif variant == 'denoising':
            self.model = DenoisingAutoencoder(64, latent_dim).to(self.device)
        elif variant == 'sparse':
            self.model = SparseAutoencoder(64, latent_dim).to(self.device)
        elif variant == 'contractive':
            self.model = ContractiveAutoencoder(64, latent_dim).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Loss function
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
        epochs: int = 10,
        batch_size: int = 128
    ) -> Dict[str, Any]:
        """Train the autoencoder variant.

        Args:
            X_train: Training data, shape (n_train, 64)
            X_test: Test data, shape (n_test, 64)
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Dictionary with training results
        """
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)

        # For denoising, we need noisy versions
        if self.variant == 'denoising':
            X_train_noisy = self._add_noise_tensor(X_train_tensor)
            X_test_noisy = self._add_noise_tensor(X_test_tensor)

        train_dataset = TensorDataset(X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for batch_x, in train_loader:
                # Prepare input based on variant
                if self.variant == 'denoising':
                    # Add noise to input
                    batch_x_noisy = self._add_noise_tensor(batch_x)
                    x_recon, z = self.model(batch_x_noisy)
                else:
                    x_recon, z = self.model(batch_x)

                # Compute reconstruction loss (always against clean target)
                recon_loss = self.criterion(x_recon, batch_x)

                # Add variant-specific regularization
                if self.variant == 'sparse':
                    # L1 penalty on latent activations
                    sparsity_loss = torch.mean(torch.abs(z))
                    loss = recon_loss + self.sparsity_weight * sparsity_loss
                elif self.variant == 'contractive':
                    # Contractive penalty (simplified version)
                    # Penalize large gradients of latent w.r.t. input
                    batch_x.requires_grad_(True)
                    z_temp = self.model.encode(batch_x)
                    dz_dx = torch.autograd.grad(
                        z_temp.sum(), batch_x, create_graph=True
                    )[0]
                    contractive_loss = torch.mean(dz_dx ** 2)
                    loss = recon_loss + self.sparsity_weight * contractive_loss
                else:
                    loss = recon_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / n_batches

            # Validation
            self.model.eval()
            with torch.no_grad():
                if self.variant == 'denoising':
                    x_recon_val, z_val = self.model(X_test_noisy)
                else:
                    x_recon_val, z_val = self.model(X_test_tensor)
                val_loss = self.criterion(x_recon_val, X_test_tensor).item()

            # Record history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)

        return {
            'training_history': self.training_history,
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1]
        }

    def _add_noise_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to tensor."""
        noise = torch.randn_like(x) * self.noise_factor
        x_noisy = torch.clamp(x + noise, 0, 1)
        return x_noisy

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode images to latent space."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            z = self.model.encode(X_tensor)
            return z.cpu().numpy()

    def reconstruct(self, X: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Reconstruct images.

        Args:
            X: Input images
            add_noise: Whether to add noise (for denoising variant demo)

        Returns:
            Reconstructed images
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if add_noise and self.variant == 'denoising':
                X_tensor = self._add_noise_tensor(X_tensor)
            x_recon, _ = self.model(X_tensor)
            return x_recon.cpu().numpy()

    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluate reconstruction quality."""
        reconstructed = self.reconstruct(X)
        mse = np.mean((X - reconstructed) ** 2)
        mae = np.mean(np.abs(X - reconstructed))

        return {
            'reconstruction_mse': float(mse),
            'avg_reconstruction_error': float(mae)
        }

    def get_learned_filters(self, n_filters: int = 16) -> np.ndarray:
        """Extract learned filters from first encoder layer.

        Args:
            n_filters: Number of filters to return

        Returns:
            Filters, shape (n_filters, 64)
        """
        # Get first layer weights (input_dim x 256)
        first_layer = self.model.encoder.fc1
        weights = first_layer.weight.data.cpu().numpy()  # Shape: (256, 64)

        # Transpose to (64, 256) and take first n_filters
        weights_t = weights.T[:, :n_filters]  # Shape: (64, n_filters)

        # Normalize for visualization
        weights_norm = (weights_t - weights_t.min(axis=0)) / (
            weights_t.max(axis=0) - weights_t.min(axis=0) + 1e-8
        )

        return weights_norm.T  # Shape: (n_filters, 64)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())

        return {
            'variant': self.variant,
            'latent_dim': self.latent_dim,
            'total_params': int(total_params),
            'encoder_params': int(encoder_params),
            'decoder_params': int(decoder_params),
            'device': str(self.device),
            'noise_factor': self.noise_factor if self.variant == 'denoising' else None,
            'sparsity_weight': self.sparsity_weight if self.variant in ['sparse', 'contractive'] else None
        }
