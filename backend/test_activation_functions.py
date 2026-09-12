#!/usr/bin/env python
"""Test script for activation functions implementation."""

import sys
sys.path.insert(0, '/home/kaitlyn/git/ai-demo/backend')

from algorithms.deep_learning.activation_functions import (
    ActivationFunctionsModel,
    ActivationFunctionsRequest,
    ActivationFunctionsResponse
)
from algorithms.deep_learning.activation_functions.model import compute_activation_functions
from algorithms.deep_learning.activation_functions.data import get_dataset_info


def test_model():
    """Test the ActivationFunctionsModel class."""
    print("Testing ActivationFunctionsModel...")

    model = ActivationFunctionsModel(alpha=0.01)

    import numpy as np
    x = np.array([-2, -1, 0, 1, 2])

    # Test ReLU
    relu_out = model.relu(x)
    print(f"  ReLU({x}) = {relu_out}")
    assert np.array_equal(relu_out, [0, 0, 0, 1, 2])

    # Test Leaky ReLU
    leaky_out = model.leaky_relu(x)
    print(f"  Leaky ReLU({x}) = {leaky_out}")
    assert leaky_out[-1] == 2  # positive values unchanged

    # Test Sigmoid
    sigmoid_out = model.sigmoid(x)
    print(f"  Sigmoid({x}) = {sigmoid_out}")
    assert all(0 < s < 1 for s in sigmoid_out)

    # Test Tanh
    tanh_out = model.tanh(x)
    print(f"  Tanh({x}) = {tanh_out}")
    assert all(-1 < t < 1 for t in tanh_out)

    print("  ✓ All activation functions working correctly")


def test_dead_neuron_demo():
    """Test the dead neuron demonstration."""
    print("\nTesting dead neuron demonstration...")

    model = ActivationFunctionsModel(alpha=0.01)
    demo = model.demonstrate_dead_neurons(x_negative=-5.0)

    print(f"  Input: {demo['input_value']}")
    print(f"  ReLU gradient: {demo['relu']['gradient']} (dead: {demo['relu']['is_dead']})")
    print(f"  Leaky ReLU gradient: {demo['leaky_relu']['gradient']} (dead: {demo['leaky_relu']['is_dead']})")

    assert demo['relu']['is_dead'] == True
    assert demo['leaky_relu']['is_dead'] == False
    assert demo['relu']['gradient'] == 0.0
    assert demo['leaky_relu']['gradient'] == 0.01

    print("  ✓ Dead neuron demonstration working correctly")


def test_comparison_data():
    """Test generation of comparison data."""
    print("\nTesting comparison data generation...")

    model = ActivationFunctionsModel(alpha=0.01)
    x, function_data = model.generate_comparison_data([-10.0, 10.0], num_points=100)

    print(f"  Generated {len(x)} points")
    print(f"  Functions: {list(function_data.keys())}")

    assert len(x) == 100
    assert len(function_data) == 6  # 6 activation functions
    assert 'relu' in function_data
    assert 'leaky_relu' in function_data

    for func_name, data in function_data.items():
        assert 'x' in data
        assert 'y' in data
        assert 'derivative' in data
        assert len(data['x']) == 100
        print(f"  ✓ {func_name}: {len(data['x'])} points")

    print("  ✓ Comparison data generation working correctly")


def test_compute_activation_functions():
    """Test the main computation function."""
    print("\nTesting compute_activation_functions...")

    result = compute_activation_functions(
        function_type='leaky_relu',
        alpha=0.01,
        input_range=[-5.0, 5.0],
        compare_all=True,
        num_points=50
    )

    print(f"  Success: {result['success']}")
    print(f"  Functions: {list(result['function_data'].keys())}")
    print(f"  Comparison table entries: {len(result['comparison_table'])}")
    print(f"  Execution time: {result['execution_time_ms']:.2f}ms")

    assert result['success'] == True
    assert len(result['function_data']) == 6
    assert len(result['comparison_table']) == 6
    assert 'dead_neuron_demo' in result
    assert 'visualization_data' in result

    print("  ✓ Computation function working correctly")


def test_request_schema():
    """Test the request schema validation."""
    print("\nTesting ActivationFunctionsRequest schema...")

    # Valid request
    request = ActivationFunctionsRequest(
        function_type='relu',
        alpha=0.01,
        input_range=[-10.0, 10.0],
        compare_all=True,
        num_points=200
    )
    print(f"  Valid request created: {request.function_type}")

    # Test with defaults
    request_defaults = ActivationFunctionsRequest()
    print(f"  Default request: function_type={request_defaults.function_type}, alpha={request_defaults.alpha}")

    assert request_defaults.function_type == 'leaky_relu'
    assert request_defaults.alpha == 0.01

    print("  ✓ Request schema working correctly")


def test_dataset_info():
    """Test dataset info function."""
    print("\nTesting dataset info...")

    info = get_dataset_info()

    print(f"  Dataset name: {info['name']}")
    print(f"  Dataset type: {info['type']}")
    print(f"  Default range: {info['default_range']}")

    assert info['name'] == 'synthetic_range'
    assert info['type'] == 'continuous'

    print("  ✓ Dataset info working correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Activation Functions Implementation Test Suite")
    print("=" * 60)

    try:
        test_model()
        test_dead_neuron_demo()
        test_comparison_data()
        test_compute_activation_functions()
        test_request_schema()
        test_dataset_info()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
