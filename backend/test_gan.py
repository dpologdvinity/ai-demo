"""Test script for GAN implementation."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from algorithms.deep_learning.gan import (
    GANModel,
    load_digits_data,
    prepare_data_for_gan
)


def test_gan_basic():
    """Test basic GAN functionality."""
    print("Testing GAN implementation...")

    # Load data
    print("\n1. Loading digits data...")
    data = load_digits_data(random_state=42)
    X = data['X']
    print(f"   Loaded {X.shape[0]} samples with {X.shape[1]} features")

    # Prepare data
    print("\n2. Preparing data for GAN...")
    X = prepare_data_for_gan(X)
    print(f"   Data range: [{X.min():.3f}, {X.max():.3f}]")

    # Initialize model
    print("\n3. Initializing GAN model...")
    model = GANModel(
        latent_dim=100,
        g_hidden=256,
        d_hidden=256,
        learning_rate=0.0002,
        random_state=42
    )
    print(f"   Model initialized on device: {model.device}")
    g_params, d_params = model.count_parameters()
    print(f"   Generator params: {g_params:,}")
    print(f"   Discriminator params: {d_params:,}")

    # Train model (small epochs for quick test)
    print("\n4. Training GAN (5 epochs for quick test)...")
    training_results = model.train(
        X=X,
        epochs=5,
        batch_size=64,
        sample_interval=2
    )

    print(f"   Training completed in {training_results['training_time_ms']:.2f}ms")
    print(f"   Loss history: {len(training_results['loss_history'])} epochs")

    # Print final losses
    final_loss = training_results['loss_history'][-1]
    print(f"\n5. Final metrics:")
    print(f"   Generator loss: {final_loss['g_loss']:.4f}")
    print(f"   Discriminator loss: {final_loss['d_loss']:.4f}")
    print(f"   D real loss: {final_loss['d_real_loss']:.4f}")
    print(f"   D fake loss: {final_loss['d_fake_loss']:.4f}")
    print(f"   D real accuracy: {final_loss['d_real_accuracy']:.4f}")
    print(f"   D fake accuracy: {final_loss['d_fake_accuracy']:.4f}")

    # Generate samples
    print("\n6. Generating samples...")
    samples = model.generate_samples(n_samples=16)
    print(f"   Generated {samples.shape[0]} samples")
    print(f"   Sample range: [{samples.min():.3f}, {samples.max():.3f}]")

    # Test latent space interpolation
    print("\n7. Testing latent space interpolation...")
    interpolated = model.interpolate_latent_space(n_steps=5, n_interpolations=2)
    print(f"   Generated {interpolated.shape[0]} interpolated samples")

    # Test decision boundary
    print("\n8. Testing decision boundary computation...")
    boundary = model.compute_decision_boundary(n_samples=50)
    print(f"   Computed boundary for {len(boundary['scores'])} samples")
    print(f"   Score range: [{min(boundary['scores']):.3f}, {max(boundary['scores']):.3f}]")

    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_gan_basic()
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
