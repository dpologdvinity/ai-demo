"""Test script for Convolutional Layers demonstration."""

import sys
import numpy as np
from algorithms.deep_learning.convolutional_layers import (
    ConvolutionalLayersModel,
    ConvolutionalLayersRequest
)
from algorithms.deep_learning.convolutional_layers.data import (
    load_sample_image,
    get_common_filters,
    get_dataset_info
)


def test_load_sample_image():
    """Test loading sample image."""
    print("Testing sample image loading...")
    image = load_sample_image()
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
    assert image.shape == (28, 28), "Image should be 28x28"
    assert image.dtype == np.float32, "Image should be float32"
    assert 0 <= image.min() and image.max() <= 1, "Image values should be in [0, 1]"
    print("  PASS\n")


def test_common_filters():
    """Test common filters."""
    print("Testing common filters...")
    filters = get_common_filters()
    print(f"  Number of filters: {len(filters)}")
    for name, kernel in filters.items():
        print(f"  - {name}: shape {kernel.shape}")
    assert 'sobel_x' in filters, "Should have Sobel X filter"
    assert 'sobel_y' in filters, "Should have Sobel Y filter"
    assert 'gaussian_blur' in filters, "Should have Gaussian blur filter"
    print("  PASS\n")


def test_dataset_info():
    """Test dataset info."""
    print("Testing dataset info...")
    info = get_dataset_info()
    print(f"  Dataset name: {info['name']}")
    print(f"  Image size: {info['image_size']}")
    print(f"  Channels: {info['channels']}")
    print(f"  Features: {len(info['features'])}")
    assert info['name'] == 'Sample Image', "Should have correct dataset name"
    print("  PASS\n")


def test_model_initialization():
    """Test model initialization with different parameters."""
    print("Testing model initialization...")

    # Test with default parameters
    model1 = ConvolutionalLayersModel()
    print(f"  Default model: {model1.num_filters} filters, kernel_size={model1.kernel_size}")

    # Test with custom parameters
    model2 = ConvolutionalLayersModel(
        num_filters=64,
        kernel_size=5,
        stride=2,
        padding='valid',
        activation='tanh'
    )
    print(f"  Custom model: {model2.num_filters} filters, kernel_size={model2.kernel_size}")
    print(f"    stride={model2.stride}, padding={model2.padding_type}, activation={model2.activation}")

    assert model1.num_filters == 32, "Default should be 32 filters"
    assert model2.num_filters == 64, "Custom should be 64 filters"
    print("  PASS\n")


def test_output_dimensions():
    """Test output dimension calculations."""
    print("Testing output dimension calculations...")

    # Test 'same' padding (output same size as input)
    model_same = ConvolutionalLayersModel(kernel_size=3, stride=1, padding='same')
    dims_same = model_same.compute_output_dimensions(28, 28)
    print(f"  Same padding (3x3, stride=1): {dims_same['height']}x{dims_same['width']}")
    assert dims_same['height'] == 28 and dims_same['width'] == 28, "Same padding should preserve size"

    # Test 'valid' padding (no padding)
    model_valid = ConvolutionalLayersModel(kernel_size=3, stride=1, padding='valid')
    dims_valid = model_valid.compute_output_dimensions(28, 28)
    print(f"  Valid padding (3x3, stride=1): {dims_valid['height']}x{dims_valid['width']}")
    assert dims_valid['height'] == 26 and dims_valid['width'] == 26, "Valid padding should reduce size"

    # Test with stride=2
    model_stride = ConvolutionalLayersModel(kernel_size=3, stride=2, padding='same')
    dims_stride = model_stride.compute_output_dimensions(28, 28)
    print(f"  Same padding (3x3, stride=2): {dims_stride['height']}x{dims_stride['width']}")
    assert dims_stride['height'] == 14 and dims_stride['width'] == 14, "Stride=2 should halve dimensions"

    print("  PASS\n")


def test_forward_pass():
    """Test forward pass through convolutional layer."""
    print("Testing forward pass...")

    model = ConvolutionalLayersModel(num_filters=8, kernel_size=3)
    image = load_sample_image()

    import torch
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    conv_out, activated_out = model.forward(image_tensor)

    print(f"  Input shape: {image_tensor.shape}")
    print(f"  Conv output shape: {conv_out.shape}")
    print(f"  Activated output shape: {activated_out.shape}")

    assert conv_out.shape[1] == 8, "Should have 8 feature maps"
    assert activated_out.shape == conv_out.shape, "Activation should preserve shape"
    print("  PASS\n")


def test_filter_extraction():
    """Test filter kernel extraction."""
    print("Testing filter kernel extraction...")

    model = ConvolutionalLayersModel(num_filters=16, kernel_size=3)
    kernels = model.extract_filter_kernels()

    print(f"  Number of kernels: {len(kernels)}")
    print(f"  Kernel shape: {kernels[0].shape}")

    assert len(kernels) == 16, "Should extract 16 kernels"
    assert kernels[0].shape == (3, 3), "Each kernel should be 3x3"
    print("  PASS\n")


def test_activation_functions():
    """Test different activation functions."""
    print("Testing activation functions...")

    import torch

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # ReLU
    model_relu = ConvolutionalLayersModel(activation='relu')
    out_relu = model_relu.apply_activation(x)
    print(f"  ReLU({x.tolist()}) = {out_relu.tolist()}")
    assert torch.all(out_relu >= 0), "ReLU should be non-negative"

    # Tanh
    model_tanh = ConvolutionalLayersModel(activation='tanh')
    out_tanh = model_tanh.apply_activation(x)
    print(f"  Tanh({x.tolist()}) = {[f'{v:.3f}' for v in out_tanh.tolist()]}")
    assert torch.all(torch.abs(out_tanh) <= 1), "Tanh should be in [-1, 1]"

    # None
    model_none = ConvolutionalLayersModel(activation='none')
    out_none = model_none.apply_activation(x)
    print(f"  None({x.tolist()}) = {out_none.tolist()}")
    assert torch.all(out_none == x), "None activation should be identity"

    print("  PASS\n")


def test_common_filters_application():
    """Test applying common filters."""
    print("Testing common filters application...")

    model = ConvolutionalLayersModel()
    image = load_sample_image()

    filtered_results = model.apply_common_filters(image)

    print(f"  Number of filters applied: {len(filtered_results)}")
    for name, result in filtered_results.items():
        print(f"  - {name}: shape {result.shape}, range [{result.min():.3f}, {result.max():.3f}]")

    assert 'sobel_x' in filtered_results, "Should have Sobel X result"
    assert 'sobel_y' in filtered_results, "Should have Sobel Y result"
    print("  PASS\n")


def test_full_demonstration():
    """Test full demonstration run."""
    print("Testing full demonstration...")

    request = ConvolutionalLayersRequest(
        num_filters=32,
        kernel_size=3,
        stride=1,
        padding='same',
        activation='relu',
        random_state=42
    )

    model = ConvolutionalLayersModel(
        num_filters=request.num_filters,
        kernel_size=request.kernel_size,
        stride=request.stride,
        padding=request.padding,
        activation=request.activation,
        random_state=request.random_state
    )

    response = model.run_demonstration(request)

    print(f"  Success: {response.success}")
    print(f"  Execution time: {response.execution_time_ms:.2f}ms")
    print(f"  Input image shape: {len(response.input_image)}x{len(response.input_image[0])}")
    print(f"  Number of filter kernels: {len(response.filter_kernels)}")
    print(f"  Number of feature maps: {len(response.feature_maps)}")
    print(f"  Output dimensions: {response.output_dimensions}")
    print(f"  Activation stats: mean={response.activation_stats['mean_activation']:.3f}, "
          f"max={response.activation_stats['max_activation']:.3f}")

    assert response.success is True, "Demonstration should succeed"
    assert len(response.filter_kernels) == 8, "Should visualize 8 filter kernels"
    assert len(response.feature_maps) == 8, "Should visualize 8 feature maps"
    assert len(response.common_filters) == 5, "Should have 5 common filters"

    print("  PASS\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CONVOLUTIONAL LAYERS DEMONSTRATION TEST")
    print("=" * 60)
    print()

    try:
        test_load_sample_image()
        test_common_filters()
        test_dataset_info()
        test_model_initialization()
        test_output_dimensions()
        test_forward_pass()
        test_filter_extraction()
        test_activation_functions()
        test_common_filters_application()
        test_full_demonstration()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\nTest failed: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
