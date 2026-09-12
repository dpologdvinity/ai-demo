"""Test script for Text Classification algorithm."""

import sys
import json
from algorithms.nlp.text_classification import (
    TextClassificationModel,
    TextClassificationParameters
)


def test_text_classification():
    """Test text classification with default parameters."""
    print("Testing Text Classification...")
    print("-" * 50)

    # Create parameters
    params = TextClassificationParameters(
        classifier_type="naive_bayes",
        max_features=1000,
        test_size=0.2,
        ngram_range=(1, 2)
    )

    print(f"Parameters:")
    print(f"  Classifier: {params.classifier_type}")
    print(f"  Max Features: {params.max_features}")
    print(f"  Test Size: {params.test_size}")
    print(f"  N-gram Range: {params.ngram_range}")
    print()

    # Train model
    print("Training model...")
    model = TextClassificationModel()
    result = model.train(params)

    if not result.success:
        print(f"ERROR: {result.error}")
        return False

    print(f"✓ Training completed in {result.execution_time_ms:.2f}ms")
    print()

    # Display results
    print("Overall Metrics:")
    for key, value in result.overall_metrics.items():
        print(f"  {key}: {value:.4f}")
    print()

    print("Per-Class Metrics:")
    for metrics in result.class_metrics:
        print(f"  {metrics.class_name}:")
        print(f"    Precision: {metrics.precision:.4f}")
        print(f"    Recall: {metrics.recall:.4f}")
        print(f"    F1-Score: {metrics.f1_score:.4f}")
        print(f"    Support: {metrics.support}")
    print()

    print(f"Total Predictions: {len(result.predictions)}")
    correct = sum(1 for p in result.predictions if p.true_label == p.predicted_label)
    print(f"Correct Predictions: {correct}/{len(result.predictions)}")
    print()

    print("Sample Predictions:")
    for i, pred in enumerate(result.sample_predictions[:5]):
        print(f"  {i+1}. Text: {pred.text[:80]}...")
        print(f"     True: {pred.true_label}, Predicted: {pred.predicted_label}")
        print(f"     Confidence: {pred.confidence:.4f}")
        print()

    print("Top Features per Class:")
    for class_name, features in result.top_features_per_class.items():
        print(f"  {class_name}:")
        top_5 = features[:5]
        feature_str = ", ".join([f"{f.feature}" for f in top_5])
        print(f"    {feature_str}")
    print()

    print("✓ Text Classification test completed successfully!")
    return True


if __name__ == "__main__":
    try:
        success = test_text_classification()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
