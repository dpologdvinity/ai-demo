#!/usr/bin/env python3
"""Test script for Gradient Descent Variants implementation."""

import sys
import json

from algorithms.deep_learning.gradient_descent import (
    GradientDescentModel,
    GradientDescentRequest,
    GradientDescentResponse
)
from algorithms.deep_learning.gradient_descent.data import get_dataset_info


def test_basic_optimization():
    """Test basic optimization with Adam."""
    print("Testing basic optimization with Adam...")

    request = GradientDescentRequest(
        optimizer_type='adam',
        learning_rate=0.1,
        momentum=0.9,
        iterations=50,
        compare_all=False,
        test_function='sphere',
        random_state=42
    )

    model = GradientDescentModel()
    response = model.run(request)

    assert response.success, "Optimization should succeed"
    assert response.single_result is not None, "Should have single result"
    assert response.single_result.final_loss < 0.1, "Should converge on sphere function"

    print(f"  ✓ Adam optimizer: final_loss={response.single_result.final_loss:.6f}")
    return response


def test_compare_all_optimizers():
    """Test comparison of all optimizers."""
    print("\nTesting comparison of all optimizers...")

    request = GradientDescentRequest(
        optimizer_type='adam',
        learning_rate=0.05,
        momentum=0.9,
        iterations=100,
        compare_all=True,
        test_function='sphere',
        random_state=42
    )

    model = GradientDescentModel()
    response = model.run(request)

    assert response.success, "Optimization should succeed"
    assert response.results is not None, "Should have results for all optimizers"
    assert len(response.results) == 5, "Should have 5 optimizers"

    print("  Results:")
    for result in response.results:
        print(f"    {result.optimizer_name:10s}: final_loss={result.final_loss:.6f}, "
              f"converged={result.iterations_to_converge is not None}")

    return response


def test_different_functions():
    """Test optimization on different test functions."""
    print("\nTesting different test functions...")

    functions = ['sphere', 'rosenbrock', 'beale', 'ackley']

    for func_name in functions:
        request = GradientDescentRequest(
            optimizer_type='adam',
            learning_rate=0.01,
            momentum=0.9,
            iterations=50,
            compare_all=False,
            test_function=func_name,
            random_state=42
        )

        model = GradientDescentModel()
        response = model.run(request)

        assert response.success, f"Optimization on {func_name} should succeed"
        print(f"  ✓ {func_name:10s}: final_loss={response.single_result.final_loss:.6f}")


def test_contour_data():
    """Test contour data generation."""
    print("\nTesting contour data generation...")

    request = GradientDescentRequest(
        optimizer_type='adam',
        learning_rate=0.1,
        iterations=50,
        compare_all=False,
        test_function='rosenbrock'
    )

    model = GradientDescentModel()
    response = model.run(request)

    assert 'x' in response.contour_data, "Should have x coordinates"
    assert 'y' in response.contour_data, "Should have y coordinates"
    assert 'z' in response.contour_data, "Should have z values"
    assert 'optimal_point' in response.contour_data, "Should have optimal point"

    print(f"  ✓ Contour data shape: {len(response.contour_data['x'])}x{len(response.contour_data['y'])}")
    print(f"  ✓ Optimal point: {response.contour_data['optimal_point']}")


def test_statistics_table():
    """Test statistics table generation."""
    print("\nTesting statistics table generation...")

    request = GradientDescentRequest(
        optimizer_type='adam',
        learning_rate=0.05,
        iterations=100,
        compare_all=True,
        test_function='sphere'
    )

    model = GradientDescentModel()
    response = model.run(request)

    assert len(response.statistics_table) > 0, "Should have statistics"

    for stat in response.statistics_table:
        assert 'optimizer' in stat, "Should have optimizer name"
        assert 'final_loss' in stat, "Should have final loss"
        assert 'path_length' in stat, "Should have path length"

    print(f"  ✓ Statistics table has {len(response.statistics_table)} entries")


def test_dataset_info():
    """Test dataset info retrieval."""
    print("\nTesting dataset info retrieval...")

    info = get_dataset_info()

    assert 'name' in info, "Should have dataset name"
    assert 'test_functions' in info, "Should have test functions list"
    assert len(info['test_functions']) > 0, "Should have at least one test function"

    print(f"  ✓ Dataset: {info['name']}")
    print(f"  ✓ Test functions: {len(info['test_functions'])}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Gradient Descent Variants - Test Suite")
    print("=" * 60)

    try:
        test_basic_optimization()
        test_compare_all_optimizers()
        test_different_functions()
        test_contour_data()
        test_statistics_table()
        test_dataset_info()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
