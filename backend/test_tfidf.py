#!/usr/bin/env python3
"""Quick test script for TF-IDF implementation."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from algorithms.nlp.tfidf import TFIDFRequest, train_tfidf

    print("Testing TF-IDF implementation...")
    print("-" * 50)

    # Create a simple test request
    request = TFIDFRequest(
        max_features=50,
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        use_idf=True,
        normalize=True
    )

    # Train model
    response = train_tfidf(request)

    if response.success:
        print("✓ TF-IDF computation successful!")
        print(f"✓ Execution time: {response.execution_time_ms:.2f}ms")
        print(f"✓ Vocabulary size: {response.metrics.get('vocabulary_size', 0)}")
        print(f"✓ Total documents: {response.metrics.get('total_documents', 0)}")
        print(f"✓ Features extracted: {len(response.feature_names)}")
        print(f"✓ Top terms per doc: {len(response.top_terms_per_doc)}")
        print(f"✓ Global top terms: {len(response.top_terms_global)}")

        print("\nTop 5 important terms globally:")
        for i, term_info in enumerate(response.top_terms_global[:5], 1):
            print(f"  {i}. {term_info['term']}: {term_info['mean_tfidf']:.4f}")

        print("\nTop 3 terms from first document:")
        if response.top_terms_per_doc:
            first_doc = response.top_terms_per_doc[0]
            print(f"  Document: {first_doc.doc_preview[:60]}...")
            for term in first_doc.top_terms[:3]:
                print(f"    - {term['term']}: {term['score']:.4f}")

        print("\n" + "="*50)
        print("All tests passed successfully!")
        print("="*50)
        sys.exit(0)
    else:
        print(f"✗ TF-IDF computation failed: {response.error}")
        sys.exit(1)

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Note: This is expected if scikit-learn is not installed.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
