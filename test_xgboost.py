#!/usr/bin/env python3
"""
Quick test script to verify XGBoost implementation.
Run this from the backend directory to test the XGBoost model.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from algorithms.ml.xgboost import XGBoostModel

def test_xgboost():
    """Test XGBoost model training."""
    print("Testing XGBoost implementation...")
    print("-" * 50)

    # Initialize model
    model = XGBoostModel()

    # Test with default parameters on wine dataset
    print("\n1. Testing with wine dataset (default parameters)...")
    result = model.train(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=6,
        subsample=1.0,
        dataset_name='wine',
        normalize=True
    )

    if result['success']:
        print("   ✓ Training successful!")
        print(f"   Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"   F1 Score: {result['metrics']['f1_score']:.4f}")
        print(f"   Execution time: {result['execution_time_ms']:.2f}ms")
        print(f"   Number of predictions: {len(result['predictions'])}")
        print(f"   Feature importance shape: {len(result['visualization_data']['feature_importance']['importance'])}")
        print(f"   Learning curves points: {len(result['visualization_data']['learning_curves']['n_estimators'])}")
    else:
        print(f"   ✗ Training failed: {result['error']}")
        return False

    # Test with iris dataset
    print("\n2. Testing with iris dataset...")
    result = model.train(
        n_estimators=100,
        learning_rate=0.2,
        max_depth=4,
        subsample=0.8,
        dataset_name='iris',
        normalize=True
    )

    if result['success']:
        print("   ✓ Training successful!")
        print(f"   Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"   Execution time: {result['execution_time_ms']:.2f}ms")
    else:
        print(f"   ✗ Training failed: {result['error']}")
        return False

    # Test with different parameters
    print("\n3. Testing with custom parameters...")
    result = model.train(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        dataset_name='wine',
        normalize=False
    )

    if result['success']:
        print("   ✓ Training successful!")
        print(f"   Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"   Execution time: {result['execution_time_ms']:.2f}ms")
    else:
        print(f"   ✗ Training failed: {result['error']}")
        return False

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
    return True

if __name__ == '__main__':
    try:
        success = test_xgboost()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
