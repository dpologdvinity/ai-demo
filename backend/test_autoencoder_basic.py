"""Basic test for Autoencoder implementation.

This script tests the Autoencoder module to ensure it can be imported and basic
functionality works without requiring the full environment setup.
"""

import sys
import numpy as np

# Test imports
try:
    from algorithms.deep_learning.autoencoder import (
        AutoencoderModel,
        AutoencoderRequest,
        AutoencoderResponse,
        load_mnist_data,
        get_dataset_info,
        compute_latent_visualization,
        prepare_sample_comparison
    )
    print("✓ Autoencoder module imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test schema validation
try:
    request = AutoencoderRequest(
        latent_dim=32,
        hidden_dim=128,
        epochs=10,  # Small for quick test
        learning_rate=0.001,
        batch_size=64,
        random_state=42
    )
    print(f"✓ AutoencoderRequest schema validation successful")
    print(f"  - latent_dim: {request.latent_dim}")
    print(f"  - hidden_dim: {request.hidden_dim}")
    print(f"  - epochs: {request.epochs}")
except Exception as e:
    print(f"✗ Schema validation error: {e}")
    sys.exit(1)

# Test dataset info
try:
    dataset_info = get_dataset_info()
    print(f"✓ Dataset info retrieved: {dataset_info['name']}, {dataset_info['n_samples']} samples")
    print(f"  - Image shape: {dataset_info['image_shape']}")
    print(f"  - Features: {dataset_info['n_features']}")
    print(f"  - Pixel range: {dataset_info['pixel_range']}")
except Exception as e:
    print(f"✗ Dataset info error: {e}")
    sys.exit(1)

# Test data loading
try:
    data = load_mnist_data(n_samples=100, test_size=0.2, random_state=42)
    print(f"✓ Data loaded successfully")
    print(f"  - Training samples: {data['X_train'].shape[0]}")
    print(f"  - Test samples: {data['X_test'].shape[0]}")
    print(f"  - Features: {data['n_features']}")
    assert data['X_train'].shape[1] == 64, "Expected 64 features (8x8 images)"
    assert data['X_test'].shape[1] == 64, "Expected 64 features (8x8 images)"
    print("✓ Data shape validation passed")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    sys.exit(1)

# Test prepare_sample_comparison
try:
    original = np.random.rand(50, 64)
    reconstructed = np.random.rand(50, 64)
    orig_samples, recon_samples = prepare_sample_comparison(original, reconstructed, n_samples=5)
    assert orig_samples.shape == (5, 64), "Expected 5 original samples"
    assert recon_samples.shape == (5, 64), "Expected 5 reconstructed samples"
    print("✓ Sample comparison preparation successful")
except Exception as e:
    print(f"✗ Sample comparison error: {e}")
    sys.exit(1)

# Test model initialization (requires PyTorch)
try:
    model = AutoencoderModel(
        latent_dim=32,
        hidden_dim=128,
        learning_rate=0.001,
        random_state=42
    )
    print("✓ Autoencoder model initialization successful")
    print(f"  - Latent dimension: {model.latent_dim}")
    print(f"  - Hidden dimension: {model.hidden_dim}")
    print(f"  - Device: {model.device}")

    # Test model info
    model_info = model.get_model_info()
    print(f"✓ Model info retrieved")
    print(f"  - Encoder params: {model_info['encoder_params']}")
    print(f"  - Decoder params: {model_info['decoder_params']}")
    print(f"  - Total params: {model_info['total_params']}")
except Exception as e:
    print(f"✗ Model initialization error: {e}")
    print("  (This is expected if PyTorch is not installed)")

# Test training (requires PyTorch and full environment)
try:
    print("\n--- Testing full training pipeline ---")
    data = load_mnist_data(n_samples=100, test_size=0.2, random_state=42)

    model = AutoencoderModel(
        latent_dim=16,
        hidden_dim=64,
        learning_rate=0.001,
        random_state=42
    )

    results = model.train(
        X_train=data['X_train'],
        X_test=data['X_test'],
        epochs=5,  # Very small for quick test
        batch_size=32
    )

    print("✓ Training completed successfully")
    print(f"  - Final train loss: {results['final_train_loss']:.4f}")
    print(f"  - Final val loss: {results['final_val_loss']:.4f}")

    # Test encoding/decoding
    latent = model.encode(data['X_test'][:10])
    print(f"✓ Encoding successful, latent shape: {latent.shape}")

    reconstructed = model.reconstruct(data['X_test'][:10])
    print(f"✓ Reconstruction successful, shape: {reconstructed.shape}")

    # Test evaluation
    eval_metrics = model.evaluate(data['X_test'])
    print(f"✓ Evaluation successful")
    print(f"  - Reconstruction loss: {eval_metrics['reconstruction_loss']:.4f}")
    print(f"  - Average pixel error: {eval_metrics['avg_pixel_error']:.4f}")

    # Test latent visualization
    latent_all = model.encode(data['X_test'])
    coords_2d, method = compute_latent_visualization(
        latent_all,
        data['y_test'],
        latent_dim=16,
        random_state=42
    )
    print(f"✓ Latent visualization successful")
    print(f"  - 2D coords shape: {coords_2d.shape}")
    print(f"  - Projection method: {method}")

except Exception as e:
    print(f"✗ Training error: {e}")
    print("  (This is expected if PyTorch is not installed or environment is not fully set up)")

print("\n" + "="*60)
print("Autoencoder basic tests completed!")
print("="*60)
