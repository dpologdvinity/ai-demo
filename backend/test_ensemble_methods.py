"""
Simple test script for Ensemble Methods implementation.
This tests the basic logic without requiring full dependencies.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from algorithms.ml.ensemble_methods.schema import (
            EnsembleMethodsRequest,
            EnsembleMethodsResponse,
            EnsembleMetrics,
            VisualizationData
        )
        print("✓ Schema imports successful")
        return True
    except Exception as e:
        print(f"✗ Schema import failed: {e}")
        return False

def test_request_schema():
    """Test request schema validation."""
    try:
        from algorithms.ml.ensemble_methods.schema import EnsembleMethodsRequest

        # Test default values
        request = EnsembleMethodsRequest()
        assert request.method == 'voting'
        assert request.n_estimators == 10
        assert request.base_model == 'decision_tree'
        assert request.max_samples == 0.8
        assert request.learning_rate == 1.0
        print("✓ Request schema defaults correct")

        # Test custom values
        request = EnsembleMethodsRequest(
            method='bagging',
            n_estimators=20,
            base_model='svm'
        )
        assert request.method == 'bagging'
        assert request.n_estimators == 20
        assert request.base_model == 'svm'
        print("✓ Request schema custom values correct")

        return True
    except Exception as e:
        print(f"✗ Request schema test failed: {e}")
        return False

def test_response_schema():
    """Test response schema structure."""
    try:
        from algorithms.ml.ensemble_methods.schema import (
            EnsembleMethodsResponse,
            EnsembleMetrics,
            VisualizationData,
            PerformanceComparisonData,
            DiversityMetrics,
            VotingData,
            SingleModelMetrics
        )

        # Create sample metrics
        metrics = EnsembleMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.93,
            f1_score=0.935,
            train_accuracy=0.98,
            test_accuracy=0.95
        )
        print("✓ EnsembleMetrics created successfully")

        # Create sample diversity metrics
        diversity = DiversityMetrics(
            disagreement=0.15,
            avg_correlation=0.65,
            q_statistic=0.45
        )
        print("✓ DiversityMetrics created successfully")

        return True
    except Exception as e:
        print(f"✗ Response schema test failed: {e}")
        return False

def test_data_module():
    """Test data module structure."""
    try:
        # Just check that the file has correct structure
        with open('algorithms/ml/ensemble_methods/data.py', 'r') as f:
            content = f.read()
            assert 'prepare_data' in content
            assert 'prepare_visualization_data' in content
            assert 'DatasetManager' in content
        print("✓ Data module structure correct")
        return True
    except Exception as e:
        print(f"✗ Data module test failed: {e}")
        return False

def test_model_module():
    """Test model module structure."""
    try:
        # Check that the file has correct structure
        with open('algorithms/ml/ensemble_methods/model.py', 'r') as f:
            content = f.read()
            assert 'EnsembleMethodsModel' in content
            assert 'BaggingClassifier' in content
            assert 'GradientBoostingClassifier' in content
            assert 'VotingClassifier' in content
            assert 'StackingClassifier' in content
            assert '_create_ensemble' in content
            assert '_calculate_diversity_metrics' in content
            assert '_get_voting_patterns' in content
        print("✓ Model module structure correct")
        return True
    except Exception as e:
        print(f"✗ Model module test failed: {e}")
        return False

def test_route_registration():
    """Test that route is properly registered."""
    try:
        with open('api/routes/ml.py', 'r') as f:
            content = f.read()
            assert 'ensemble-methods' in content
            assert 'EnsembleMethodsModel' in content
            assert 'train_ensemble_methods' in content
            assert 'get_ensemble_methods_info' in content
            assert 'ensemble_methods_metadata' in content
        print("✓ Route registration correct")
        return True
    except Exception as e:
        print(f"✗ Route registration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Ensemble Methods Implementation")
    print("=" * 50)

    tests = [
        test_imports,
        test_request_schema,
        test_response_schema,
        test_data_module,
        test_model_module,
        test_route_registration
    ]

    results = []
    for test in tests:
        results.append(test())
        print()

    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
