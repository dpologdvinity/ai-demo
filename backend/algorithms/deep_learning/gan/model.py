"""Generative Adversarial Network (GAN) implementation using PyTorch.

This module provides a GAN implementation for generating digit-like images.
The GAN consists of two neural networks: a Generator that creates fake images
from random noise, and a Discriminator that tries to distinguish real from fake images.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time


class Generator(nn.Module):
    """Generator network that creates fake images from random noise.

    The generator takes a random noise vector (latent vector) as input and
    transforms it through hidden layers to produce an image.

    Architecture:
        Input (latent_dim) -> Hidden (g_hidden) with LeakyReLU -> Output (64) with Sigmoid

    Attributes:
        model: Sequential neural network model
    """

    def __init__(self, latent_dim: int = 100, hidden_size: int = 128, output_size: int = 64):
        """Initialize Generator network.

        Args:
            latent_dim: Dimension of the input noise vector. Default: 100
            hidden_size: Size of the hidden layer. Default: 128
            output_size: Size of the output image (64 for 8x8). Default: 64
        """
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator.

        Args:
            z: Random noise tensor, shape (batch_size, latent_dim)

        Returns:
            Generated images, shape (batch_size, 64)
        """
        return self.model(z)


class Discriminator(nn.Module):
    """Discriminator network that classifies images as real or fake.

    The discriminator takes an image as input and outputs a probability
    that the image is real (not generated).

    Architecture:
        Input (64) -> Hidden (d_hidden) with LeakyReLU -> Output (1) with Sigmoid

    Attributes:
        model: Sequential neural network model
    """

    def __init__(self, input_size: int = 64, hidden_size: int = 128):
        """Initialize Discriminator network.

        Args:
            input_size: Size of the input image (64 for 8x8). Default: 64
            hidden_size: Size of the hidden layer. Default: 128
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Probability output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator.

        Args:
            x: Image tensor, shape (batch_size, 64)

        Returns:
            Probability that image is real, shape (batch_size, 1)
        """
        return self.model(x)


class GANModel:
    """Generative Adversarial Network for generating digit-like images.

    This class implements a GAN that learns to generate 8x8 grayscale images
    resembling handwritten digits. The Generator and Discriminator are trained
    adversarially using binary cross-entropy loss.

    Attributes:
        generator: Generator network
        discriminator: Discriminator network
        latent_dim: Dimension of the noise vector
        device: Device to run computations on (CPU or CUDA)
        training_time_ms: Time taken to train in milliseconds
    """

    def __init__(
        self,
        latent_dim: int = 100,
        g_hidden: int = 128,
        d_hidden: int = 128,
        learning_rate: float = 0.0002,
        random_state: int = 42
    ):
        """Initialize GAN model.

        Args:
            latent_dim: Dimension of the noise vector input to generator.
                Default: 100
            g_hidden: Size of generator's hidden layer. Default: 128
            d_hidden: Size of discriminator's hidden layer. Default: 128
            learning_rate: Learning rate for both networks. Default: 0.0002
            random_state: Random seed for reproducibility. Default: 42

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if latent_dim < 32 or latent_dim > 256:
            raise ValueError(f"latent_dim must be between 32 and 256, got {latent_dim}")
        if g_hidden < 128 or g_hidden > 512:
            raise ValueError(f"g_hidden must be between 128 and 512, got {g_hidden}")
        if d_hidden < 128 or d_hidden > 512:
            raise ValueError(f"d_hidden must be between 128 and 512, got {d_hidden}")

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, g_hidden, 64).to(self.device)
        self.discriminator = Discriminator(64, d_hidden).to(self.device)

        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999)
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Training state
        self.training_time_ms = 0.0

    def train(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        sample_interval: int = 10
    ) -> Dict[str, Any]:
        """Train the GAN on digit images.

        Trains both Generator and Discriminator adversarially. The Discriminator
        learns to distinguish real from fake images, while the Generator learns
        to fool the Discriminator.

        Args:
            X: Training data, shape (n_samples, 64). Images should be normalized
               to [0, 1] range.
            epochs: Number of training epochs. Default: 100
            batch_size: Batch size for training. Default: 64
            sample_interval: Interval (in epochs) to save generated samples.
                Default: 10

        Returns:
            Dictionary containing:
                - loss_history: List of loss dictionaries per epoch
                - generated_samples: List of generated samples at intervals
                - training_time_ms: Training time in milliseconds

        Raises:
            ValueError: If X is empty or has wrong shape.
        """
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if X.ndim != 2 or X.shape[1] != 64:
            raise ValueError(f"X must have shape (n_samples, 64), got {X.shape}")

        start_time = time.time()

        # Prepare data loader
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Real and fake labels
        real_label = 1.0
        fake_label = 0.0

        # Training history
        loss_history = []
        generated_samples = []

        # Training loop
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_d_real_loss = 0.0
            epoch_d_fake_loss = 0.0
            epoch_d_real_acc = 0.0
            epoch_d_fake_acc = 0.0
            n_batches = 0

            for batch_idx, (real_images,) in enumerate(dataloader):
                current_batch_size = real_images.size(0)

                # Labels
                real_labels = torch.full(
                    (current_batch_size, 1),
                    real_label,
                    dtype=torch.float,
                    device=self.device
                )
                fake_labels = torch.full(
                    (current_batch_size, 1),
                    fake_label,
                    dtype=torch.float,
                    device=self.device
                )

                # ---------------------
                # Train Discriminator
                # ---------------------
                self.discriminator.zero_grad()

                # Real images
                real_output = self.discriminator(real_images)
                d_real_loss = self.criterion(real_output, real_labels)

                # Calculate real accuracy (correctly classified as real)
                d_real_pred = (real_output > 0.5).float()
                d_real_accuracy = (d_real_pred == real_labels).float().mean()

                # Fake images
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                fake_images = self.generator(z)
                fake_output = self.discriminator(fake_images.detach())
                d_fake_loss = self.criterion(fake_output, fake_labels)

                # Calculate fake accuracy (correctly classified as fake)
                d_fake_pred = (fake_output > 0.5).float()
                d_fake_accuracy = (d_fake_pred == fake_labels).float().mean()

                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()

                # ---------------------
                # Train Generator
                # ---------------------
                self.generator.zero_grad()

                # Generate fake images and get discriminator's opinion
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                fake_images = self.generator(z)
                fake_output = self.discriminator(fake_images)

                # Generator tries to fool discriminator (wants fake_output close to 1)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()

                # Accumulate losses and accuracies
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_d_real_loss += d_real_loss.item()
                epoch_d_fake_loss += d_fake_loss.item()
                epoch_d_real_acc += d_real_accuracy.item()
                epoch_d_fake_acc += d_fake_accuracy.item()
                n_batches += 1

            # Average losses and accuracies for this epoch
            avg_g_loss = epoch_g_loss / n_batches
            avg_d_loss = epoch_d_loss / n_batches
            avg_d_real_loss = epoch_d_real_loss / n_batches
            avg_d_fake_loss = epoch_d_fake_loss / n_batches
            avg_d_real_acc = epoch_d_real_acc / n_batches
            avg_d_fake_acc = epoch_d_fake_acc / n_batches

            # Record loss history
            loss_history.append({
                'epoch': epoch,
                'g_loss': float(avg_g_loss),
                'd_loss': float(avg_d_loss),
                'd_real_loss': float(avg_d_real_loss),
                'd_fake_loss': float(avg_d_fake_loss),
                'd_real_accuracy': float(avg_d_real_acc),
                'd_fake_accuracy': float(avg_d_fake_acc)
            })

            # Save generated samples at intervals
            if epoch % sample_interval == 0 or epoch == epochs - 1:
                samples = self.generate_samples(n_samples=16)
                generated_samples.append({
                    'epoch': epoch,
                    'samples': samples.tolist()
                })

        self.training_time_ms = (time.time() - start_time) * 1000

        return {
            'loss_history': loss_history,
            'generated_samples': generated_samples,
            'training_time_ms': self.training_time_ms
        }

    def generate_samples(self, n_samples: int = 16) -> np.ndarray:
        """Generate fake images using the trained generator.

        Args:
            n_samples: Number of samples to generate. Default: 16

        Returns:
            Generated images, shape (n_samples, 64)
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(z)
            samples = samples.cpu().numpy()
        self.generator.train()
        return samples

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.

        Returns:
            Dictionary containing model architecture and parameter counts.
        """
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())

        return {
            'latent_dim': self.latent_dim,
            'generator_params': int(g_params),
            'discriminator_params': int(d_params),
            'total_params': int(g_params + d_params),
            'device': str(self.device)
        }

    def count_parameters(self) -> Tuple[int, int]:
        """Count trainable parameters in both networks.

        Returns:
            Tuple of (generator_params, discriminator_params)
        """
        g_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        return int(g_params), int(d_params)

    def interpolate_latent_space(
        self,
        n_steps: int = 10,
        n_interpolations: int = 5
    ) -> np.ndarray:
        """Generate images by interpolating in latent space.

        Creates smooth transitions between random points in latent space
        to demonstrate the continuity of the learned distribution.

        Args:
            n_steps: Number of interpolation steps between each pair. Default: 10
            n_interpolations: Number of interpolation sequences to generate. Default: 5

        Returns:
            Array of interpolated images, shape (n_interpolations * n_steps, 64)
        """
        self.generator.eval()
        all_interpolations = []

        with torch.no_grad():
            for _ in range(n_interpolations):
                # Sample two random points in latent space
                z1 = torch.randn(1, self.latent_dim, device=self.device)
                z2 = torch.randn(1, self.latent_dim, device=self.device)

                # Create interpolation steps
                alphas = torch.linspace(0, 1, n_steps, device=self.device)
                interpolations = []

                for alpha in alphas:
                    # Linear interpolation: z = (1-α)z1 + αz2
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    img = self.generator(z_interp)
                    interpolations.append(img.cpu().numpy())

                all_interpolations.extend(interpolations)

        self.generator.train()
        return np.vstack(all_interpolations)

    def get_discriminator_scores(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """Get discriminator scores for given images.

        Args:
            X: Images to score, shape (n_samples, 64)

        Returns:
            Discriminator scores (probability of being real), shape (n_samples,)
        """
        self.discriminator.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            scores = self.discriminator(X_tensor)
            scores = scores.cpu().numpy().flatten()
        self.discriminator.train()
        return scores

    def compute_decision_boundary(
        self,
        n_samples: int = 200
    ) -> Dict[str, Any]:
        """Compute discriminator decision boundary visualization data.

        Generates samples in latent space, produces images, and gets
        discriminator scores to visualize decision boundaries.

        Args:
            n_samples: Number of samples to generate. Default: 200

        Returns:
            Dictionary containing:
                - latent_vectors: 2D projection of latent vectors for visualization
                - scores: Discriminator scores for each generated image
                - images: Generated images
        """
        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():
            # Generate random latent vectors
            z = torch.randn(n_samples, self.latent_dim, device=self.device)

            # Generate images
            generated_images = self.generator(z)

            # Get discriminator scores
            scores = self.discriminator(generated_images)

            # Convert to numpy
            z_np = z.cpu().numpy()
            images_np = generated_images.cpu().numpy()
            scores_np = scores.cpu().numpy().flatten()

            # Project latent vectors to 2D for visualization (using first 2 dimensions)
            # For better visualization, we could use PCA, but using first 2 dims is simpler
            latent_2d = z_np[:, :2]

        self.generator.train()
        self.discriminator.train()

        return {
            'latent_vectors': latent_2d.tolist(),
            'scores': scores_np.tolist(),
            'images': images_np.tolist()
        }
