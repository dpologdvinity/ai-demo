#!/usr/bin/env python
"""Validation script for activation functions integration."""

import sys
sys.path.insert(0, '/home/kaitlyn/git/ai-demo/backend')


def validate_module_structure():
    """Validate module file structure."""
    print("Validating module structure...")

    import os
    base_path = '/home/kaitlyn/git/ai-demo/backend/algorithms/deep_learning/activation_functions'

    required_files = [
        '__init__.py',
        'model.py',
        'schema.py',
        'data.py',
        'README.md'
    ]

    for file in required_files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} MISSING!")
            return False

    return True


def validate_imports():
    """Validate all imports work correctly."""
    print("\nValidating imports...")

    try:
        from algorithms.deep_learning.activation_functions import (
            ActivationFunctionsModel,
            ActivationFunctionsRequest,
            ActivationFunctionsResponse
        )
        print("  ✓ Module imports")

        from algorithms.deep_learning.activation_functions.model import compute_activation_functions
        print("  ✓ Function imports")

        from algorithms.deep_learning.activation_functions.data import get_dataset_info
        print("  ✓ Data imports")

        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def validate_route_registration():
    """Validate route registration in deep_learning.py."""
    print("\nValidating route registration...")

    try:
        # Read the routes file
        with open('/home/kaitlyn/git/ai-demo/backend/api/routes/deep_learning.py', 'r') as f:
            content = f.read()

        # Check for import
        if 'from algorithms.deep_learning.activation_functions import' in content:
            print("  ✓ Import statement present")
        else:
            print("  ✗ Import statement missing")
            return False

        # Check for metadata registration
        if 'activation_functions_metadata = AlgorithmMetadata' in content:
            print("  ✓ Metadata definition present")
        else:
            print("  ✗ Metadata definition missing")
            return False

        if 'AlgorithmRegistry.register(activation_functions_metadata)' in content:
            print("  ✓ Metadata registration present")
        else:
            print("  ✗ Metadata registration missing")
            return False

        # Check for endpoints
        if '@router.post("/activation-functions/compute"' in content:
            print("  ✓ POST /activation-functions/compute endpoint present")
        else:
            print("  ✗ POST endpoint missing")
            return False

        if '@router.get("/activation-functions/info"' in content:
            print("  ✓ GET /activation-functions/info endpoint present")
        else:
            print("  ✗ GET endpoint missing")
            return False

        return True
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return False


def validate_metadata():
    """Validate algorithm metadata."""
    print("\nValidating algorithm metadata...")

    try:
        from utils.algorithm_metadata import AlgorithmRegistry

        # Import routes to trigger registration
        import api.routes.deep_learning

        metadata = AlgorithmRegistry.get('activation-functions')

        if not metadata:
            print("  ✗ Metadata not registered")
            return False

        print(f"  ✓ Metadata registered")
        print(f"    - Name: {metadata.name}")
        print(f"    - Slug: {metadata.slug}")
        print(f"    - Category: {metadata.category}")
        print(f"    - Difficulty: {metadata.difficulty}")
        print(f"    - Parameters: {len(metadata.parameters)}")
        print(f"    - Tags: {', '.join(metadata.tags)}")

        # Validate parameters
        expected_params = ['function_type', 'alpha', 'input_range', 'compare_all', 'num_points']
        actual_params = [p.name for p in metadata.parameters]

        for param in expected_params:
            if param in actual_params:
                print(f"    ✓ Parameter '{param}' present")
            else:
                print(f"    ✗ Parameter '{param}' missing")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Metadata validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_functionality():
    """Validate core functionality."""
    print("\nValidating core functionality...")

    try:
        from algorithms.deep_learning.activation_functions.model import compute_activation_functions

        result = compute_activation_functions(
            function_type='leaky_relu',
            alpha=0.01,
            input_range=[-5.0, 5.0],
            compare_all=True,
            num_points=50
        )

        if not result['success']:
            print("  ✗ Computation failed")
            return False

        print(f"  ✓ Computation successful")
        print(f"    - Functions: {len(result['function_data'])}")
        print(f"    - Comparison entries: {len(result['comparison_table'])}")
        print(f"    - Execution time: {result['execution_time_ms']:.2f}ms")

        # Validate output structure
        required_keys = ['success', 'function_data', 'comparison_table',
                        'dead_neuron_demo', 'visualization_data', 'execution_time_ms']

        for key in required_keys:
            if key in result:
                print(f"    ✓ Key '{key}' present")
            else:
                print(f"    ✗ Key '{key}' missing")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Functionality validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_schema():
    """Validate Pydantic schemas."""
    print("\nValidating Pydantic schemas...")

    try:
        from algorithms.deep_learning.activation_functions import (
            ActivationFunctionsRequest,
            ActivationFunctionsResponse
        )

        # Test request schema with valid data
        request = ActivationFunctionsRequest(
            function_type='relu',
            alpha=0.01,
            input_range=[-10.0, 10.0],
            compare_all=True,
            num_points=200
        )
        print(f"  ✓ Request schema valid")
        print(f"    - function_type: {request.function_type}")
        print(f"    - alpha: {request.alpha}")
        print(f"    - num_points: {request.num_points}")

        # Test with defaults
        request_defaults = ActivationFunctionsRequest()
        print(f"  ✓ Request defaults work")
        print(f"    - default function_type: {request_defaults.function_type}")
        print(f"    - default alpha: {request_defaults.alpha}")

        # Test validation
        try:
            invalid_request = ActivationFunctionsRequest(function_type='invalid')
            print(f"  ✗ Validation not working (should reject invalid function_type)")
            return False
        except:
            print(f"  ✓ Validation working (rejects invalid function_type)")

        return True
    except Exception as e:
        print(f"  ✗ Schema validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("Activation Functions Integration Validation")
    print("=" * 70)

    checks = [
        ("Module Structure", validate_module_structure),
        ("Imports", validate_imports),
        ("Route Registration", validate_route_registration),
        ("Algorithm Metadata", validate_metadata),
        ("Core Functionality", validate_functionality),
        ("Pydantic Schemas", validate_schema),
    ]

    results = []

    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} | {name}")

    print("=" * 70)

    all_passed = all(result for _, result in results)

    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED!")
        print("\nThe activation functions module is fully integrated and ready to use.")
        print("\nEndpoints:")
        print("  - POST /deep-learning/activation-functions/compute")
        print("  - GET  /deep-learning/activation-functions/info")
        return 0
    else:
        print("✗ SOME VALIDATION CHECKS FAILED!")
        print("\nPlease review the failures above and fix the issues.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
