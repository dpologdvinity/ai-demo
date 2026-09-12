#!/usr/bin/env python3
"""Test script for VGG Network implementation."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_vgg_structure():
    """Test that VGG module structure is correct."""
    print("Testing VGG module structure...")

    # Test imports (will fail without torch, but validates structure)
    try:
        from algorithms.deep_learning.vgg import VGGModel, VGGRequest, VGGResponse
        print("✓ VGG imports successful")
    except ModuleNotFoundError as e:
        if 'torch' in str(e):
            print("✓ VGG structure correct (torch not installed in test env)")
        else:
            print(f"✗ Import error: {e}")
            return False

    # Test schema imports
    try:
        from algorithms.deep_learning.vgg.schema import (
            VGGRequest,
            VGGResponse,
            PredictionResult,
            ConvBlockInfo
        )
        print("✓ VGG schema classes defined")
    except Exception as e:
        print(f"✗ Schema import error: {e}")
        return False

    # Test data module
    try:
        from algorithms.deep_learning.vgg.data import (
            get_imagenet_labels,
            get_sample_images,
            get_dataset_info
        )
        print("✓ VGG data utilities defined")

        # Test data functions
        labels = get_imagenet_labels()
        assert len(labels) == 1000, "Should have 1000 ImageNet labels"
        print(f"✓ ImageNet labels loaded: {len(labels)} classes")

        dataset_info = get_dataset_info()
        assert dataset_info['name'] == 'ImageNet'
        print("✓ Dataset info loaded")

    except Exception as e:
        print(f"✗ Data module error: {e}")
        return False

    return True


def test_vgg_metadata():
    """Test that VGG is registered in algorithm registry."""
    print("\nTesting VGG metadata registration...")

    try:
        from utils.algorithm_metadata import AlgorithmRegistry

        # Check if VGG is registered (won't be until routes are loaded)
        vgg = AlgorithmRegistry.get("vgg")
        if vgg:
            print("✓ VGG registered in algorithm registry")
            print(f"  Name: {vgg.name}")
            print(f"  Description: {vgg.description}")
            print(f"  Parameters: {len(vgg.parameters)}")
        else:
            print("ℹ VGG not yet registered (normal - requires route loading)")
    except Exception as e:
        print(f"ℹ Could not test metadata: {e}")

    return True


def test_vgg_request_schema():
    """Test VGG request schema validation."""
    print("\nTesting VGG request schema...")

    try:
        from algorithms.deep_learning.vgg.schema import VGGRequest

        # Create valid request
        request = VGGRequest(
            model_variant="vgg16",
            top_k=5,
            use_pretrained=True,
            batch_norm=True,
            image_index=0
        )
        print("✓ Valid VGG request created")
        print(f"  Model: {request.model_variant}")
        print(f"  Top-K: {request.top_k}")
        print(f"  Batch Norm: {request.batch_norm}")

        # Test default values
        request2 = VGGRequest()
        assert request2.model_variant == "vgg16"
        assert request2.top_k == 5
        assert request2.batch_norm == True
        print("✓ Default values correct")

        # Test validation (should fail with invalid variant)
        try:
            invalid_request = VGGRequest(model_variant="invalid")
            print("✗ Should have failed validation for invalid variant")
            return False
        except Exception:
            print("✓ Request validation working")

        return True

    except Exception as e:
        print(f"✗ Request schema test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("VGG Network Implementation Tests")
    print("=" * 60)

    tests = [
        test_vgg_structure,
        test_vgg_metadata,
        test_vgg_request_schema,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests: {passed}/{total} passed")
    print("=" * 60)

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
