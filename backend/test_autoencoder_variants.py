"""Test script for Autoencoder Variants implementation."""

import sys
import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from algorithms.deep_learning.autoencoder_variants import (
            AutoencoderVariantsModel,
            VanillaAutoencoder,
            DenoisingAutoencoder,
            SparseAutoencoder,
            ContractiveAutoencoder,
            AutoencoderVariantsRequest,
            AutoencoderVariantsResponse,
            load_mnist_data,
            get_dataset_info,
            compute_latent_visualization,
            prepare_variant_comparison,
            add_noise
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_schema_validation():
    """Test Pydantic schema validation."""
    print("\nTesting schema validation...")
    try:
        from algorithms.deep_learning.autoencoder_variants import AutoencoderVariantsRequest

        # Test with valid parameters
        request = AutoencoderVariantsRequest(
            variant='vanilla',
            latent_dim=32,
            epochs=10,
            learning_rate=0.001,
            noise_factor=0.3,
            sparsity_weight=0.001,
            batch_size=128,
            random_state=42
        )
        print(f"✓ Schema validation successful")
        print(f"  - variant: {request.variant}")
        print(f"  - latent_dim: {request.latent_dim}")
        print(f"  - epochs: {request.epochs}")
        return True
    except Exception as e:
        print(f"✗ Schema validation error: {e}")
        return False


def test_dataset_loading():
    """Test dataset loading."""
    print("\nTesting dataset loading...")
    try:
        from algorithms.deep_learning.autoencoder_variants import load_mnist_data, get_dataset_info

        dataset_info = get_dataset_info()
        print(f"✓ Dataset info retrieved: {dataset_info['name']}")
        print(f"  - n_samples: {dataset_info['n_samples']}")
        print(f"  - image_shape: {dataset_info['image_shape']}")

        data = load_mnist_data(n_samples=100, test_size=0.2, random_state=42)
        print(f"✓ Data loaded successfully")
        print(f"  - Train samples: {data['X_train'].shape[0]}")
        print(f"  - Test samples: {data['X_test'].shape[0]}")
        print(f"  - Features: {data['n_features']}")
        return True
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        return False


def test_noise_addition():
    """Test noise addition utility."""
    print("\nTesting noise addition...")
    try:
        from algorithms.deep_learning.autoencoder_variants import add_noise

        X = np.random.rand(10, 64)
        X_noisy = add_noise(X, noise_factor=0.3)

        assert X_noisy.shape == X.shape, "Noisy data shape mismatch"
        assert np.all(X_noisy >= 0) and np.all(X_noisy <= 1), "Noisy data out of range"
        print(f"✓ Noise addition successful")
        print(f"  - Original range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  - Noisy range: [{X_noisy.min():.3f}, {X_noisy.max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ Noise addition error: {e}")
        return False


def test_model_initialization():
    """Test model initialization for all variants."""
    print("\nTesting model initialization...")
    try:
        from algorithms.deep_learning.autoencoder_variants import AutoencoderVariantsModel

        variants = ['vanilla', 'denoising', 'sparse', 'contractive']
        for variant in variants:
            model = AutoencoderVariantsModel(
                variant=variant,
                latent_dim=32,
                learning_rate=0.001,
                random_state=42
            )
            model_info = model.get_model_info()
            print(f"✓ {variant.capitalize()} autoencoder initialized")
            print(f"  - Total params: {model_info['total_params']}")
            print(f"  - Device: {model_info['device']}")

        return True
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training():
    """Test training on small dataset."""
    print("\nTesting training (vanilla autoencoder)...")
    try:
        from algorithms.deep_learning.autoencoder_variants import (
            AutoencoderVariantsModel,
            load_mnist_data
        )

        # Load small dataset
        data = load_mnist_data(n_samples=100, test_size=0.2, random_state=42)

        # Train vanilla autoencoder
        model = AutoencoderVariantsModel(
            variant='vanilla',
            latent_dim=16,
            learning_rate=0.001,
            random_state=42
        )

        results = model.train(
            X_train=data['X_train'],
            X_test=data['X_test'],
            epochs=3,  # Small for quick test
            batch_size=32
        )

        print(f"✓ Training completed")
        print(f"  - Final train loss: {results['final_train_loss']:.4f}")
        print(f"  - Final val loss: {results['final_val_loss']:.4f}")

        # Test encoding/decoding
        latent = model.encode(data['X_test'][:10])
        print(f"✓ Encoding successful, latent shape: {latent.shape}")

        reconstructed = model.reconstruct(data['X_test'][:10])
        print(f"✓ Reconstruction successful, shape: {reconstructed.shape}")

        # Test evaluation
        metrics = model.evaluate(data['X_test'])
        print(f"✓ Evaluation successful")
        print(f"  - Reconstruction MSE: {metrics['reconstruction_mse']:.4f}")

        # Test learned filters
        filters = model.get_learned_filters(n_filters=8)
        print(f"✓ Filter extraction successful, shape: {filters.shape}")

        return True
    except Exception as e:
        print(f"✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_latent_visualization():
    """Test latent space visualization."""
    print("\nTesting latent visualization...")
    try:
        from algorithms.deep_learning.autoencoder_variants import (
            AutoencoderVariantsModel,
            load_mnist_data,
            compute_latent_visualization
        )

        data = load_mnist_data(n_samples=100, test_size=0.2, random_state=42)

        model = AutoencoderVariantsModel(
            variant='vanilla',
            latent_dim=32,
            random_state=42
        )

        # Get latent representations
        latent = model.encode(data['X_test'])

        # Compute 2D visualization
        coords_2d, method = compute_latent_visualization(
            latent,
            data['y_test'],
            latent_dim=32,
            random_state=42
        )

        print(f"✓ Latent visualization successful")
        print(f"  - 2D coords shape: {coords_2d.shape}")
        print(f"  - Projection method: {method}")

        return True
    except Exception as e:
        print(f"✗ Latent visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Autoencoder Variants Test Suite")
    print("="*60)

    tests = [
        test_imports,
        test_schema_validation,
        test_dataset_loading,
        test_noise_addition,
        test_model_initialization,
        test_training,
        test_latent_visualization
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "="*60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("="*60)

    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
