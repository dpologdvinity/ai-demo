#!/usr/bin/env python3
"""Test script for GloVe implementation."""

import sys
import json

# Test 1: Import the GloVe module
print("=" * 60)
print("Test 1: Importing GloVe module")
print("=" * 60)
try:
    from algorithms.nlp.glove import GloVeModel, GloVeParameters
    print("✓ Successfully imported GloVe classes")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Create a query and test the model
print("\n" + "=" * 60)
print("Test 2: Testing GloVe query")
print("=" * 60)
try:
    params = GloVeParameters(
        embedding_dim=100,
        query_word="king",
        top_k=10,
        analogy_word_a="king",
        analogy_word_b="man",
        analogy_word_c="woman"
    )
    print(f"✓ Created parameters: {params.model_dump()}")

    model = GloVeModel()
    print("✓ Created GloVe model")

    result = model.query(params)
    print(f"✓ Query completed: success={result.success}")

    if result.success:
        print(f"\nMetrics:")
        print(f"  - Vocabulary size: {result.metrics.get('vocab_size', 0)}")
        print(f"  - Embedding dimension: {result.metrics.get('embedding_dim', 0)}")
        print(f"  - Execution time: {result.execution_time_ms:.2f}ms")

        print(f"\nSimilar words to '{params.query_word}':")
        for i, word in enumerate(result.similar_words[:5], 1):
            print(f"  {i}. {word.word}: {word.similarity:.4f}")

        if result.analogy_result:
            print(f"\nWord Analogy:")
            print(f"  Query: {result.analogy_result.query}")
            print(f"  Result: {result.analogy_result.result_word} (confidence: {result.analogy_result.similarity:.4f})")
            print(f"  Top 5 results:")
            for i, word in enumerate(result.analogy_result.top_results[:5], 1):
                print(f"    {i}. {word.word}: {word.similarity:.4f}")

        print(f"\nVisualization data:")
        print(f"  - 2D embeddings: {len(result.embeddings_2d)} words")
        print(f"  - Cosine similarity matrix: {result.cosine_similarity_matrix.get('shape', 'N/A')}")
        print(f"  - Vocabulary sample: {len(result.vocabulary_sample)} words")

    else:
        print(f"✗ Query failed: {result.error}")
        sys.exit(1)

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test with different parameters
print("\n" + "=" * 60)
print("Test 3: Testing with different words")
print("=" * 60)
try:
    test_words = ["computer", "paris", "happy", "dog"]
    for word in test_words:
        params = GloVeParameters(query_word=word, top_k=5)
        model = GloVeModel()
        result = model.query(params)

        if result.success and result.similar_words:
            similar = [f"{w.word}({w.similarity:.2f})" for w in result.similar_words[:3]]
            print(f"  '{word}': {', '.join(similar)}")
        elif result.success:
            print(f"  '{word}': [not in vocabulary]")
        else:
            print(f"  '{word}': [error]")
    print("✓ Multiple queries successful")
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
