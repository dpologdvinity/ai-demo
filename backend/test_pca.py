#!/usr/bin/env python3
"""Quick test script for PCA implementation."""

import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pca_imports():
    """Test that PCA modules can be imported."""
    try:
        from algorithms.ml.pca import PCAModel, PCARequest, PCAResponse
        print("✓ PCA imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_pca_model():
    """Test PCA model with sample data."""
    try:
        import numpy as np
        from algorithms.ml.pca.model import PCAModel

        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 10)

        # Initialize and fit PCA
        model = PCAModel(n_components=2)
        result = model.fit_transform(X)

        print("✓ PCA model training successful")
        print(f"  - Transformed shape: {result['transformed_data'].shape}")
        print(f"  - Variance explained: {sum(result['explained_variance_ratio']):.2%}")
        return True
    except Exception as e:
        print(f"✗ Model test error: {e}")
        return False


def test_pca_endpoint():
    """Test PCA with digits dataset."""
    try:
        from algorithms.ml.pca.model import PCAModel
        from algorithms.ml.pca.data import load_digits_data

        # Load data
        data = load_digits_data()
        X_train = data['X_train']

        # Fit PCA
        model = PCAModel(n_components=3)
        result = model.fit_transform(X_train)

        print("✓ PCA with digits dataset successful")
        print(f"  - Original features: {X_train.shape[1]}")
        print(f"  - Reduced to: {result['n_components']} components")
        print(f"  - Training samples: {X_train.shape[0]}")
        print(f"  - Variance explained by PC1: {result['explained_variance_ratio'][0]:.2%}")
        print(f"  - Variance explained by PC2: {result['explained_variance_ratio'][1]:.2%}")
        print(f"  - Variance explained by PC3: {result['explained_variance_ratio'][2]:.2%}")
        print(f"  - Cumulative variance: {result['cumulative_variance_ratio'][-1]:.2%}")
        return True
    except Exception as e:
        print(f"✗ Endpoint test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing PCA Implementation")
    print("=" * 60)

    tests = [
        test_pca_imports,
        test_pca_model,
        test_pca_endpoint,
    ]

    results = []
    for test in tests:
        print(f"\n{test.__name__}:")
        results.append(test())

    print("\n" + "=" * 60)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print("=" * 60)

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
