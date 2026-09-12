"""Simple quick test for Word2Vec implementation."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.nlp.word2vec import Word2VecModel, Word2VecParameters


def test_word2vec_basic():
    """Test basic Word2Vec training without expensive operations."""
    print("Testing Word2Vec implementation (basic)...")
    print("-" * 50)

    # Create parameters
    params = Word2VecParameters(
        vector_size=50,  # Minimum allowed
        window=3,
        min_count=1,
        sg=0,  # CBOW
        epochs=5,  # Fewer epochs
        random_state=42
    )

    print(f"Parameters: vector_size={params.vector_size}, window={params.window}")
    print("-" * 50)

    # Create and train model
    model = Word2VecModel()
    print("Training Word2Vec model...")

    try:
        # Just build vocabulary and co-occurrence matrix
        if params.use_custom_corpus and params.custom_corpus:
            from algorithms.nlp.word2vec.data import prepare_custom_corpus
            model.corpus = prepare_custom_corpus(params.custom_corpus)
        else:
            from algorithms.nlp.word2vec.data import get_default_corpus
            model.corpus = get_default_corpus()

        model.parameters = params

        print(f"Corpus loaded: {len(model.corpus)} sentences")

        # Build vocabulary
        model._build_vocabulary()
        print(f"Vocabulary built: {len(model.vocabulary)} words")

        # Build co-occurrence matrix
        print("Building co-occurrence matrix...")
        cooccurrence_matrix = model._build_cooccurrence_matrix()
        print(f"Co-occurrence matrix shape: {cooccurrence_matrix.shape}")

        # Compute embeddings
        print("Computing embeddings...")
        model._compute_embeddings(cooccurrence_matrix)
        print(f"Embeddings computed: {len(model.word_vectors)} word vectors")

        # Test similar words
        if 'machine' in model.word_vectors:
            print("\nTesting similarity search for 'machine':")
            similar = model.find_similar_words('machine', topn=5)
            for word, score in similar:
                print(f"  {word}: {score:.4f}")

        print("\n✓ Basic training successful!")
        return True

    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_word2vec_basic()

    if success:
        print("\n" + "="*50)
        print("✓ Word2Vec basic test passed!")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("✗ Test failed")
        print("="*50)
        sys.exit(1)
