"""Test script for Feature Importance API endpoint."""

import sys
import warnings
warnings.filterwarnings('ignore')

# Test import of the route
try:
    from api.routes.ml import router
    print("✓ ML router imported successfully")
except Exception as e:
    print(f"✗ Failed to import router: {e}")
    sys.exit(1)

# Test metadata registration
try:
    from utils.algorithm_metadata import AlgorithmRegistry
    metadata = AlgorithmRegistry.get('feature-importance')

    if metadata:
        print(f"✓ Feature Importance metadata registered")
        print(f"  - Name: {metadata.name}")
        print(f"  - Slug: {metadata.slug}")
        print(f"  - Category: {metadata.category}")
        print(f"  - Difficulty: {metadata.difficulty}")
        print(f"  - Parameters: {len(metadata.parameters)} defined")
        print(f"  - Use cases: {len(metadata.use_cases)} listed")

        # List parameters
        print(f"\n  Parameters:")
        for param in metadata.parameters:
            print(f"    - {param.name} ({param.type}): {param.description[:50]}...")
    else:
        print("✗ Metadata not found in registry")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error checking metadata: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test the actual implementation
try:
    from algorithms.ml.feature_importance import FeatureImportanceModel, FeatureImportanceRequest

    print(f"\n✓ Feature Importance module imported")

    # Quick test
    request = FeatureImportanceRequest(
        method='tree',
        model_type='random_forest',
        dataset='housing',
        n_estimators=10,  # Small for speed
        top_k=3
    )

    model = FeatureImportanceModel()
    print(f"  - Running quick test with {request.model_type} on {request.dataset}...")

    response = model.train(request)

    print(f"  - Training completed in {response.execution_time_ms:.2f}ms")
    print(f"  - Methods: {list(response.feature_importance.keys())}")
    print(f"  - Model metrics: R2={response.metrics.get('r2_score', 0):.3f}")
    print(f"  - Top feature: {response.feature_rankings['tree'][0]['feature']}")

except Exception as e:
    print(f"✗ Implementation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! Feature Importance is ready.")
