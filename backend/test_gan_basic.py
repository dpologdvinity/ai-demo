"""Basic test for GAN implementation.

This script tests the GAN module to ensure it can be imported and basic
functionality works without requiring the full environment setup.
"""

import sys
import numpy as np

# Test imports
try:
    from algorithms.deep_learning.gan import (
        GANModel,
        GANRequest,
        GANResponse,
        load_digits_data,
        get_dataset_info,
        prepare_data_for_gan
    )
    print("✓ GAN module imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test schema validation
try:
    request = GANRequest(
        latent_dim=100,
        g_hidden=128,
        d_hidden=128,
        learning_rate=0.0002,
        epochs=5,  # Small for quick test
        batch_size=64,
        random_state=42
    )
    print(f"✓ GANRequest schema validation successful: {request.model_dump()}")
except Exception as e:
    print(f"✗ Schema validation error: {e}")
    sys.exit(1)

# Test dataset info
try:
    dataset_info = get_dataset_info()
    print(f"✓ Dataset info retrieved: {dataset_info['name']}, {dataset_info['n_samples']} samples")
except Exception as e:
    print(f"✗ Dataset info error: {e}")
    sys.exit(1)

# Test data loading
try:
    data = load_digits_data(random_state=42)
    print(f"✓ Data loaded: shape {data['X'].shape}, features {data['n_features']}")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    sys.exit(1)

# Test data preparation
try:
    X_prepared = prepare_data_for_gan(data['X'])
    print(f"✓ Data prepared: min={X_prepared.min():.3f}, max={X_prepared.max():.3f}")
except Exception as e:
    print(f"✗ Data preparation error: {e}")
    sys.exit(1)

# Test model initialization
try:
    model = GANModel(
        latent_dim=100,
        g_hidden=128,
        d_hidden=128,
        learning_rate=0.0002,
        random_state=42
    )
    model_info = model.get_model_info()
    print(f"✓ Model initialized: {model_info['generator_params']} generator params, "
          f"{model_info['discriminator_params']} discriminator params")
except Exception as e:
    print(f"✗ Model initialization error: {e}")
    sys.exit(1)

# Test sample generation (before training)
try:
    samples = model.generate_samples(n_samples=4)
    print(f"✓ Sample generation successful: shape {samples.shape}")
except Exception as e:
    print(f"✗ Sample generation error: {e}")
    sys.exit(1)

# Test training with very small dataset and few epochs
try:
    print("\nTesting training with small dataset (this may take a moment)...")
    X_small = data['X'][:100]  # Use only 100 samples for quick test
    training_results = model.train(
        X=X_small,
        epochs=2,  # Only 2 epochs for quick test
        batch_size=32,
        sample_interval=1
    )
    print(f"✓ Training successful: {len(training_results['loss_history'])} epochs recorded")
    print(f"  Final G loss: {training_results['loss_history'][-1]['g_loss']:.4f}")
    print(f"  Final D loss: {training_results['loss_history'][-1]['d_loss']:.4f}")
    print(f"  Training time: {training_results['training_time_ms']:.2f}ms")
    print(f"  Generated samples: {len(training_results['generated_samples'])} checkpoints")
except Exception as e:
    print(f"✗ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test final sample generation
try:
    final_samples = model.generate_samples(n_samples=16)
    print(f"✓ Final sample generation: shape {final_samples.shape}")
except Exception as e:
    print(f"✗ Final sample generation error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed! GAN implementation is working correctly.")
print("="*60)
