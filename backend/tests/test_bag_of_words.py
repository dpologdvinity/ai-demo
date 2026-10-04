"""Smoke tests for Bag of Words algorithm."""

from algorithms.nlp.bag_of_words.model import BagOfWordsModel
from algorithms.nlp.bag_of_words.schema import BagOfWordsRequest


def test_bow_model_initialization():
    """Test Bag of Words model can be initialized."""
    model = BagOfWordsModel()
    assert model is not None


def test_bow_vectorize_with_defaults():
    """Test Bag of Words vectorize method with default parameters."""
    model = BagOfWordsModel()
    request = BagOfWordsRequest(
        documents=["hello world", "foo bar"],
        max_features=100,
        min_df=1,
        max_df=1.0,
        binary=False
    )

    response = model.vectorize(request)

    assert response is not None
    assert hasattr(response, 'success')
    assert response.success is True
    assert hasattr(response, 'document_term_matrix')
