"""Test that text classification module can be imported."""

print("Testing text classification imports...")

try:
    from algorithms.nlp.text_classification import (
        TextClassificationModel,
        TextClassificationParameters,
        TextClassificationResponse,
        get_default_dataset,
        get_category_descriptions
    )
    print("✓ All imports successful")

    # Test dataset
    texts, labels = get_default_dataset()
    print(f"✓ Default dataset loaded: {len(texts)} samples")

    # Test categories
    categories = get_category_descriptions()
    print(f"✓ Categories loaded: {list(categories.keys())}")

    # Test parameter creation
    params = TextClassificationParameters()
    print(f"✓ Default parameters created: {params.classifier_type}")

    print("\nAll tests passed!")

except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
