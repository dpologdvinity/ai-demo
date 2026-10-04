"""Smoke tests for Word2Vec algorithm."""

from algorithms.nlp.word2vec.model import Word2VecModel
from algorithms.nlp.word2vec.schema import Word2VecParameters


def test_word2vec_model_initialization():
    """Test Word2Vec model can be initialized."""
    model = Word2VecModel()
    assert model is not None


def test_word2vec_train_with_defaults():
    """Test Word2Vec train method with default parameters."""
    model = Word2VecModel()
    params = Word2VecParameters(
        vector_size=100,
        window=5,
        min_count=1,
        sg=0,
        epochs=5
    )

    response = model.train(params)

    assert response is not None
    assert hasattr(response, 'success')
    assert response.success is True
    assert hasattr(response, 'embeddings_2d')
