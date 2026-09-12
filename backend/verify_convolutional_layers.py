"""Verification script for Convolutional Layers implementation."""

import os
import sys


def check_file_exists(filepath, description):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists


def check_imports():
    """Check if modules import successfully."""
    print("\nChecking imports...")
    try:
        from algorithms.deep_learning.convolutional_layers import (
            ConvolutionalLayersModel,
            ConvolutionalLayersRequest,
            ConvolutionalLayersResponse
        )
        print("✓ Main module imports successfully")

        from algorithms.deep_learning.convolutional_layers.data import (
            load_sample_image,
            get_common_filters,
            get_dataset_info
        )
        print("✓ Data module imports successfully")

        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def check_functionality():
    """Check basic functionality."""
    print("\nChecking basic functionality...")
    try:
        from algorithms.deep_learning.convolutional_layers import ConvolutionalLayersModel
        from algorithms.deep_learning.convolutional_layers.data import load_sample_image

        # Test model creation
        model = ConvolutionalLayersModel(num_filters=8, kernel_size=3)
        print(f"✓ Model created: {model.num_filters} filters, kernel_size={model.kernel_size}")

        # Test image loading
        image = load_sample_image()
        print(f"✓ Sample image loaded: shape={image.shape}, dtype={image.dtype}")

        # Test forward pass
        import torch
        img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        conv_out, activated = model.forward(img_tensor)
        print(f"✓ Forward pass successful: output shape={conv_out.shape}")

        return True
    except Exception as e:
        print(f"✗ Functionality error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run verification checks."""
    print("=" * 70)
    print("CONVOLUTIONAL LAYERS IMPLEMENTATION VERIFICATION")
    print("=" * 70)

    all_passed = True

    # Check files
    print("\nChecking files...")
    base_path = "algorithms/deep_learning/convolutional_layers"

    files_to_check = [
        (f"{base_path}/__init__.py", "Module __init__"),
        (f"{base_path}/schema.py", "Schema definitions"),
        (f"{base_path}/model.py", "Model implementation"),
        (f"{base_path}/data.py", "Data utilities"),
        ("test_convolutional_layers.py", "Unit tests"),
        ("test_convolutional_layers_api.py", "API tests"),
        ("CONVOLUTIONAL_LAYERS_IMPLEMENTATION.md", "Documentation"),
    ]

    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_passed = False

    # Check route registration
    print("\nChecking API route registration...")
    route_file = "api/routes/deep_learning.py"
    if os.path.exists(route_file):
        with open(route_file, 'r') as f:
            content = f.read()

        checks = [
            ("convolutional_layers_metadata", "Metadata registration"),
            ("ConvolutionalLayersRequest", "Request schema import"),
            ("ConvolutionalLayersResponse", "Response schema import"),
            ("/convolutional-layers/demo", "Demo endpoint"),
            ("/convolutional-layers/info", "Info endpoint"),
        ]

        for search_str, description in checks:
            if search_str in content:
                print(f"✓ {description} found")
            else:
                print(f"✗ {description} NOT FOUND")
                all_passed = False
    else:
        print(f"✗ Route file not found: {route_file}")
        all_passed = False

    # Check imports
    if not check_imports():
        all_passed = False

    # Check functionality
    if not check_functionality():
        all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nImplementation is complete and ready to use.")
        print("\nNext steps:")
        print("1. Start the backend server: uvicorn main:app --reload")
        print("2. Test the API: python test_convolutional_layers_api.py")
        print("3. Access in browser: http://localhost:8000/docs")
        print("4. Look for 'convolutional-layers' endpoints")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease review the errors above.")

    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
