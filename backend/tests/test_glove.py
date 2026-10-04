"""Smoke tests for GloVe Word Embeddings algorithm."""

from algorithms.nlp.glove.model import GloVeModel
from algorithms.nlp.glove.schema import GloVeParameters


def test_glove_model_initialization():
    """Test GloVe model can be initialized."""
    model = GloVeModel()
    assert model is not None


def test_glove_query_with_defaults():
    """Test GloVe query method with default parameters."""
    model = GloVeModel()
    params = GloVeParameters(
        embedding_dim=100,
        top_k=10,
        query_word="king",
        analogy_word_a="king",
        analogy_word_b="man",
        analogy_word_c="woman"
    )

    response = model.query(params)

    assert response is not None
    assert hasattr(response, 'similar_words')
    assert isinstance(response.similar_words, list)
