"""Simple test script for Word2Vec implementation."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.nlp.word2vec import Word2VecModel, Word2VecParameters


def test_word2vec_training():
    """Test Word2Vec model training."""
    print("Testing Word2Vec implementation...")
    print("-" * 50)

    # Create parameters
    params = Word2VecParameters(
        vector_size=100,
        window=5,
        min_count=2,  # Lower min_count for small corpus
        sg=0,  # CBOW
        epochs=10,
        random_state=42
    )

    print(f"Parameters: {params.model_dump()}")
    print("-" * 50)

    # Create and train model
    model = Word2VecModel()
    print("Training Word2Vec model...")
    result = model.train(params)

    # Check results
    if result.success:
        print("\n✓ Training successful!")
        print(f"Execution time: {result.execution_time_ms:.2f}ms")
        print(f"\nMetrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")

        print(f"\nVocabulary sample (first 10 words):")
        for word in result.vocabulary_sample[:10]:
            print(f"  - {word}")

        print(f"\nSimilar words examples:")
        for query_word, similar_words in result.similar_words.items():
            print(f"  '{query_word}' is similar to:")
            for sw in similar_words:
                print(f"    - {sw.word} (similarity: {sw.similarity:.4f})")

        print(f"\nWord analogies:")
        for analogy in result.analogies:
            print(f"  {analogy.query} ≈ {analogy.result} (similarity: {analogy.similarity:.4f})")

        print(f"\n2D embeddings generated: {len(result.embeddings_2d)} words")
        if result.embeddings_2d:
            print(f"  Example: '{result.embeddings_2d[0]['word']}' at ({result.embeddings_2d[0]['x']:.2f}, {result.embeddings_2d[0]['y']:.2f})")

        print("\n✓ All tests passed!")
        return True
    else:
        print(f"\n✗ Training failed: {result.error}")
        return False


def test_custom_corpus():
    """Test Word2Vec with custom corpus."""
    print("\n" + "="*50)
    print("Testing Word2Vec with custom corpus...")
    print("="*50)

    custom_text = """
    Python is a great programming language for machine learning.
    Machine learning algorithms can learn from data.
    Neural networks are powerful machine learning models.
    Deep learning uses neural networks with many layers.
    Natural language processing helps computers understand text.
    Text mining extracts useful information from documents.
    """

    params = Word2VecParameters(
        vector_size=50,
        window=3,
        min_count=1,
        sg=1,  # Skip-gram
        epochs=15,
        use_custom_corpus=True,
        custom_corpus=custom_text,
        random_state=42
    )

    print(f"Custom corpus length: {len(custom_text)} characters")
    print("-" * 50)

    model = Word2VecModel()
    print("Training with custom corpus...")
    result = model.train(params)

    if result.success:
        print("\n✓ Custom corpus training successful!")
        print(f"Vocabulary size: {result.metrics['vocab_size']}")
        print(f"Training algorithm: {result.metrics['training_algorithm']}")
        return True
    else:
        print(f"\n✗ Custom corpus training failed: {result.error}")
        return False


if __name__ == "__main__":
    # Test default corpus
    success1 = test_word2vec_training()

    # Test custom corpus
    success2 = test_custom_corpus()

    if success1 and success2:
        print("\n" + "="*50)
        print("✓ All Word2Vec tests passed!")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("✗ Some tests failed")
        print("="*50)
        sys.exit(1)
